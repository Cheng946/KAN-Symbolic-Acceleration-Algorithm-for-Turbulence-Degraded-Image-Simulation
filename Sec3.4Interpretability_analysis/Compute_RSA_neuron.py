import os
import math
import argparse
import joblib
import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import correlate
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

import efficient_kan.kan as E_kan
import MyKANnetLoader_2


# ========================== 初等函数配置 ==========================

SUPPORTED_ELEMENTARY_FUNCTIONS = {
    "silu": lambda x: torch.nn.functional.silu(x),
    "relu": lambda x: torch.nn.functional.relu(x),
    "sin": lambda x: torch.sin(x),
    "cos": lambda x: torch.cos(x),
    "exp": lambda x: torch.exp(torch.clamp(x, -10, 10)),
    "log": lambda x: torch.log(torch.abs(x) + 1e-6),
    "tanh": lambda x: torch.tanh(x),
    "sigmoid": lambda x: torch.sigmoid(x),
    "sqrt": lambda x: torch.sqrt(torch.abs(x) + 1e-6),
    "square": lambda x: torch.square(x),
    "abs": lambda x: torch.abs(x),
    "identity": lambda x: x,
}

DEFAULT_ELEMENTARY_FUNCTIONS = ["silu", "relu", "tanh", "sigmoid", "abs", "identity"]


# ========================== SymbolicKAN 定义 ==========================

class SymbolicKANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale_base=1.0,
        scale_mlp=1.0,
        base_activation=torch.nn.SiLU,
        elementary_functions=None,
    ):
        super(SymbolicKANLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale_base = scale_base
        self.scale_mlp = scale_mlp
        self.base_activation = base_activation()

        if elementary_functions is None:
            self.elementary_functions = DEFAULT_ELEMENTARY_FUNCTIONS
        else:
            self.elementary_functions = elementary_functions

        self.num_ef = len(self.elementary_functions)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.ef_mlp_linears = nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(out_features, in_features))
            for _ in range(self.num_ef)
        ])

        self.ef_mlp_biases = nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(out_features))
            for _ in range(self.num_ef)
        ])

        self.ef_weights = torch.nn.Parameter(torch.ones(self.num_ef))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        for i in range(self.num_ef):
            torch.nn.init.kaiming_uniform_(self.ef_mlp_linears[i], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.ef_mlp_linears[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.ef_mlp_biases[i], -bound, bound)

        torch.nn.init.constant_(self.ef_weights, 1.0 / self.num_ef)

    def apply_elementary_function(self, x, func_idx):
        func_name = self.elementary_functions[func_idx]
        mlp_output = nn.functional.linear(
            x,
            self.ef_mlp_linears[func_idx],
            self.ef_mlp_biases[func_idx],
        )
        ef_output = SUPPORTED_ELEMENTARY_FUNCTIONS[func_name](mlp_output)
        return ef_output * self.ef_weights[func_idx]

    def forward(self, x):
        assert x.size(-1) == self.in_features

        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = nn.functional.linear(self.base_activation(x), self.base_weight)

        mlp_outputs = []
        for i in range(self.num_ef):
            mlp_outputs.append(self.apply_elementary_function(x, i))

        mlp_output = torch.stack(mlp_outputs, dim=-1).sum(dim=-1) * self.scale_mlp

        output = base_output + mlp_output
        return output.reshape(*original_shape[:-1], self.out_features)


class SymbolicKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        scale_base=1.0,
        scale_mlp=1.0,
        base_activation=torch.nn.SiLU,
        elementary_functions=None,
    ):
        super(SymbolicKAN, self).__init__()

        self.layers = torch.nn.ModuleList()

        if elementary_functions is None:
            ef_list = DEFAULT_ELEMENTARY_FUNCTIONS
        else:
            ef_list = elementary_functions

        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                SymbolicKANLinear(
                    in_features=in_features,
                    out_features=out_features,
                    scale_base=scale_base,
                    scale_mlp=scale_mlp,
                    base_activation=base_activation,
                    elementary_functions=ef_list,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ========================== 物理前向 ==========================

def load_zernike_poly(path):
    z = np.load(path)
    z = np.transpose(z, (2, 0, 1))
    return z[3:, :, :].astype(np.float64)


def func_a2phase(a, zern_poly):
    phase = np.sum(a[:, None, None] * zern_poly, axis=0)
    return phase


def func_phase2exp(phase):
    return np.exp(1j * 2 * np.pi * phase)


def func_exp2wave(exp_phase, N=128):
    x, y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    mask = (np.sqrt(x ** 2 + y ** 2) <= 1).astype(float)

    wave = mask * exp_phase

    p = np.sum(np.abs(wave) ** 2)
    wave = wave * (((1 / (N ** 2)) / p) ** 0.5)

    return wave


def func_wave2otf(wave, N=128):
    otf = correlate(wave, wave, mode="same") * (N ** 2)
    otf = otf[::-1, ::-1]
    return otf


# ========================== RSA 工具 ==========================

def compute_RDM(representations):
    dist_vec = pdist(representations, metric="euclidean")
    return squareform(dist_vec)


def compute_complex_RDM(complex_representations):
    Z = np.asarray(complex_representations)
    N = Z.shape[0]
    Z = Z.reshape(N, -1).astype(np.complex64)

    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / (norm + 1e-12)

    S = np.abs(Z @ Z.conj().T)
    S = np.clip(S, 0.0, 1.0)

    rdm = 1.0 - S
    np.fill_diagonal(rdm, 0.0)

    return rdm.astype(np.float32)


def rsa_correlation(rdm_a, rdm_b):
    mask = np.triu(np.ones_like(rdm_a, dtype=bool), k=1)

    vec_a = rdm_a[mask]
    vec_b = rdm_b[mask]

    r, p = pearsonr(vec_a, vec_b)
    return r, p


def compute_neuron_RDM(neuron_act):
    x = neuron_act.reshape(-1, 1)
    return compute_RDM(x)


# ========================== 网络层激活提取 ==========================

def get_all_layer_activations(net, x_batch, device):
    activations = []

    def hook_fn(module, inp, out):
        activations.append(out.detach().cpu())

    hooks = []
    for layer in net.layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    net.eval()
    with torch.no_grad():
        _ = net(x_batch.to(device))

    for h in hooks:
        h.remove()

    return activations


# ========================== Neuron-level RSA 主函数 ==========================

def neuron_level_rsa(layer_repr, phys_rdm_map, layer_name):
    """
    layer_repr: [N, hidden_dim]
    phys_rdm_map: dict[str, RDM]
    """

    results = []

    hidden_dim = layer_repr.shape[1]

    for neuron_idx in range(hidden_dim):
        neuron_act = layer_repr[:, neuron_idx]

        if np.std(neuron_act) < 1e-12:
            continue

        rdm_neuron = compute_neuron_RDM(neuron_act)

        scores = {}
        for phys_name, rdm_phys in phys_rdm_map.items():
            r, p = rsa_correlation(rdm_phys, rdm_neuron)
            scores[phys_name] = {
                "r": float(r),
                "p": float(p),
            }

        best_phys = max(scores.keys(), key=lambda k: scores[k]["r"])
        best_r = scores[best_phys]["r"]
        best_p = scores[best_phys]["p"]

        result = {
            "layer": layer_name,
            "neuron": neuron_idx,
            "best_phys": best_phys,
            "best_r": best_r,
            "best_p": best_p,
        }

        for phys_name in phys_rdm_map.keys():
            result[f"r_{phys_name}"] = scores[phys_name]["r"]
            result[f"p_{phys_name}"] = scores[phys_name]["p"]

        results.append(result)

    return results


def save_results_csv(results, save_path):
    import pandas as pd

    df = pd.DataFrame(results)
    df = df.sort_values(["layer", "best_r"], ascending=[True, False])
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"Neuron-level RSA CSV saved to: {save_path}")
    return df


def plot_label_distribution(df, save_path):
    import pandas as pd

    count_df = df.groupby(["layer", "best_phys"]).size().reset_index(name="count")

    layers = sorted(df["layer"].unique())
    labels = sorted(df["best_phys"].unique())

    mat = np.zeros((len(layers), len(labels)))

    for i, layer in enumerate(layers):
        for j, label in enumerate(labels):
            val = count_df[
                (count_df["layer"] == layer) &
                (count_df["best_phys"] == label)
            ]["count"]
            mat[i, j] = int(val.iloc[0]) if len(val) > 0 else 0

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(mat, cmap="Blues")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(layers)

    ax.set_xlabel("Best Physical Representation")
    ax.set_ylabel("Network Layer")
    ax.set_title("Neuron-level RSA Label Distribution")

    for i in range(len(layers)):
        for j in range(len(labels)):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, label="Neuron Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Label distribution heatmap saved to: {save_path}")


def plot_top_neurons(df, save_path, top_k=30):
    top_df = df.sort_values("best_r", ascending=False).head(top_k)

    names = [
        f"{row['layer']}-N{int(row['neuron'])}\n{row['best_phys']}"
        for _, row in top_df.iterrows()
    ]

    values = top_df["best_r"].values

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), names, rotation=75, ha="right")
    plt.ylabel("Best RSA Pearson r")
    plt.title(f"Top {top_k} Physical-aligned Neurons")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Top neurons plot saved to: {save_path}")


# ========================== 主程序 ==========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/aiofm/F/20250723_k-fold-cross-validation-KAN/1-fold",
    )

    parser.add_argument("--batchSize", type=int, default=5120)

    parser.add_argument(
        "--zernike_poly_path",
        type=str,
        default="36—128ZernPoly.npy",
    )

    parser.add_argument(
        "--pca_model_path",
        type=str,
        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/New_pca_model-70.pkl",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/Last_SymbolicKAN_Para_Silu_1_fold_4L_-524-524-524_15_2.pt",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_RSA/neuron_level_rsa",
    )

    parser.add_argument("--max_rsa_samples", type=int, default=2000)
    parser.add_argument("--forward_batch", type=int, default=512)
    parser.add_argument("--r_threshold", type=float, default=0.1)

    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ====================== 加载模型 ======================

    zern_poly = load_zernike_poly(opt.zernike_poly_path)

    layers_hidden = [33, 524, 524, 524, 70]

    net = SymbolicKAN(
        layers_hidden=layers_hidden,
        elementary_functions=DEFAULT_ELEMENTARY_FUNCTIONS,
    ).to(device)

    checkpoint = torch.load(opt.model_path, map_location=device)
    net.load_state_dict(checkpoint["parameter"])

    net.eval()

    # ====================== 加载测试集 ======================

    train_iter, val_iter, test_iter = MyKANnetLoader_2.load_dataset(opt)

    a_list = []
    alpha_list = []

    for batch in test_iter:
        a, alpha = batch
        a_list.append(a)
        alpha_list.append(alpha)

    a_test = torch.cat(a_list, dim=0).numpy().reshape(-1, 33)
    alpha_test = torch.cat(alpha_list, dim=0).numpy().reshape(-1, 70)

    N_test = a_test.shape[0]

    print("Full test set a_test:", a_test.shape)
    print("Full test set alpha_test:", alpha_test.shape)

    # ====================== RSA采样 ======================

    if N_test > opt.max_rsa_samples:
        np.random.seed(0)
        sample_idx = np.random.choice(N_test, size=opt.max_rsa_samples, replace=False)
        a_rsa = a_test[sample_idx]
        print(f"Sample {opt.max_rsa_samples} / {N_test} samples for Neuron-level RSA")
    else:
        a_rsa = a_test

    N_rsa = a_rsa.shape[0]
    grid_size = 128
    phase_dim = grid_size * grid_size

    # ====================== 计算物理表征 ======================

    print("\n===== Calculating physical representations =====")

    phase_repr = np.zeros((N_rsa, phase_dim), dtype=np.float32)
    sin_phase_repr = np.zeros((N_rsa, phase_dim), dtype=np.float32)
    cos_phase_repr = np.zeros((N_rsa, phase_dim), dtype=np.float32)

    exp_phase_repr = np.zeros((N_rsa, grid_size, grid_size), dtype=np.complex64)
    wave_repr = np.zeros((N_rsa, grid_size, grid_size), dtype=np.complex64)
    otf_repr = np.zeros((N_rsa, grid_size, grid_size), dtype=np.complex64)

    for i in range(N_rsa):
        a_samp = a_rsa[i]

        phase = func_a2phase(a_samp, zern_poly)
        exp_phase = func_phase2exp(phase)
        wave = func_exp2wave(exp_phase, N=grid_size)
        otf = func_wave2otf(wave, N=grid_size)

        phase_repr[i] = phase.ravel()
        sin_phase_repr[i] = np.sin(2 * np.pi * phase).ravel()
        cos_phase_repr[i] = np.cos(2 * np.pi * phase).ravel()

        exp_phase_repr[i] = exp_phase.astype(np.complex64)
        wave_repr[i] = wave.astype(np.complex64)
        otf_repr[i] = otf.astype(np.complex64)

    # ====================== 计算物理RDM ======================

    print("\n===== Computing physical RDMs =====")

    phys_rdm_map = {
        "Phase": compute_RDM(phase_repr),
        "SinPhase": compute_RDM(sin_phase_repr),
        "CosPhase": compute_RDM(cos_phase_repr),
        "ExpPhase": compute_complex_RDM(exp_phase_repr),
        "Wavefront": compute_complex_RDM(wave_repr),
        "OTF": compute_complex_RDM(otf_repr),
    }

    # ====================== 提取网络层激活 ======================

    print("\n===== Extracting SymbolicKAN layer activations =====")

    h_lists = [[], [], [], []]

    for start in range(0, N_rsa, opt.forward_batch):
        end = min(start + opt.forward_batch, N_rsa)

        batch_np = a_rsa[start:end]
        batch_t = torch.from_numpy(batch_np).float()

        acts = get_all_layer_activations(net, batch_t, device)

        for i in range(4):
            h_lists[i].append(acts[i].numpy())

    h1_repr = np.concatenate(h_lists[0], axis=0)
    h2_repr = np.concatenate(h_lists[1], axis=0)
    h3_repr = np.concatenate(h_lists[2], axis=0)
    h4_repr = np.concatenate(h_lists[3], axis=0)

    layer_repr_map = {
        "Layer1": h1_repr,
        "Layer2": h2_repr,
        "Layer3": h3_repr,
        "Layer4": h4_repr,
    }

    print("Layer1:", h1_repr.shape)
    print("Layer2:", h2_repr.shape)
    print("Layer3:", h3_repr.shape)
    print("Layer4:", h4_repr.shape)

    # ====================== Neuron-level RSA ======================

    print("\n===== Running Neuron-level RSA =====")

    all_results = []

    for layer_name, layer_repr in layer_repr_map.items():
        print(f"\n--- {layer_name} ---")

        layer_results = neuron_level_rsa(
            layer_repr=layer_repr,
            phys_rdm_map=phys_rdm_map,
            layer_name=layer_name,
        )

        all_results.extend(layer_results)

        top_results = sorted(layer_results, key=lambda x: x["best_r"], reverse=True)[:20]

        for item in top_results:
            print(
                f"{layer_name} Neuron {item['neuron']:4d} "
                f"-> {item['best_phys']:10s}, "
                f"r = {item['best_r']:.4f}, "
                f"p = {item['best_p']:.2e}"
            )

    # ====================== 保存结果 ======================

    csv_path = os.path.join(opt.save_dir, "neuron_level_rsa_results.csv")
    df = save_results_csv(all_results, csv_path)

    # 只统计显著/有效对应神经元
    df_valid = df[df["best_r"] >= opt.r_threshold].copy()

    csv_valid_path = os.path.join(opt.save_dir, f"neuron_level_rsa_results_r_ge_{opt.r_threshold}.csv")
    df_valid.to_csv(csv_valid_path, index=False, encoding="utf-8-sig")

    print(f"Filtered CSV saved to: {csv_valid_path}")
    print(f"Valid neurons with r >= {opt.r_threshold}: {len(df_valid)} / {len(df)}")

    # ====================== 画图 ======================

    dist_plot_path = os.path.join(opt.save_dir, "neuron_label_distribution.png")
    plot_label_distribution(df_valid, dist_plot_path)

    top_plot_path = os.path.join(opt.save_dir, "top_physical_aligned_neurons.png")
    plot_top_neurons(df, top_plot_path, top_k=30)

    print("\n===== Done =====")


if __name__ == "__main__":
    main()