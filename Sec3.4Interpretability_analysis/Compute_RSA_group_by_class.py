# -*- coding: utf-8 -*-
"""
Group-vector RSA after neuron-level physical-label classification.

功能：
1. 读取前一步 neuron-level RSA 生成的分类 CSV：
   - 默认需要列：layer, neuron, best_phys
   - 可选列：best_r，用于再次按阈值筛选节点
2. 重新加载测试集、模型，提取各层激活。
3. 在每一层内，把同一 best_phys 类别的节点合并成一个群体向量：
   group_repr = layer_repr[:, neuron_indices]，shape = [N_samples, N_group_neurons]
4. 对每个群体向量计算 RDM，再与各物理量 RDM 做 RSA。
   - OTF 的 RDM 采用复数实部/虚部拼接：[real(OTF), imag(OTF)]。
5. 用 Distance = 1 - RSA 表示“该群体向量”与各物理量之间的距离。

输出：
- group_vector_rsa_results.csv：长表，每行是 Layer + 类别节点群体向量
- group_vector_distance_matrix.csv：距离矩阵，行是 Layer/类别群体，列是物理量
- group_vector_rsa_matrix.csv：RSA 相似度矩阵
- group_vector_neuron_members.csv：每个群体包含哪些神经元
- group_vector_distance_heatmap.png：1 - RSA 距离热图
- group_vector_rsa_heatmap.png：RSA 热图

示例：
python Compute_RSA_group_by_class.py \
  --data_root /media/aiofm/F/20250723_k-fold-cross-validation-KAN/1-fold \
  --zernike_poly_path 36—128ZernPoly.npy \
  --pca_model_path /home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/New_pca_model-70.pkl \
  --model_path /home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/Last_SymbolicKAN_Para_Silu_1_fold_4L_-524-524-524_15_2.pt \
  --classification_csv /home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_RSA/neuron_level_rsa/neuron_level_rsa_results_r_ge_0.1.csv \
  --save_dir /home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_RSA/group_vector_rsa \
  --max_rsa_samples 2000 \
  --forward_batch 512
"""

import os
import math
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import correlate
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr

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


# ========================== RSA / RDM 工具 ==========================

def zscore_features(x, eps=1e-12):
    """对表征矩阵的每个特征维度做 z-score，避免某些节点尺度过大主导欧氏 RDM。"""
    x = np.asarray(x, dtype=np.float32)
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def compute_RDM(representations, metric="euclidean", zscore=False):
    x = np.asarray(representations)
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if zscore:
        x = zscore_features(x)

    if x.shape[0] < 2:
        raise ValueError("RDM 至少需要 2 个样本。")

    dist_vec = pdist(x, metric=metric)
    return squareform(dist_vec).astype(np.float32)


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


def compute_complex_realimag_concat_RDM(
    complex_representations,
    metric="euclidean",
    zscore=False,
):
    """
    将复数表征展平后拆成实部和虚部，并在特征维拼接：
        X = [Re(Z), Im(Z)]
    然后用普通实数特征矩阵计算 RDM。

    这里用于 OTF 的 RDM，避免使用 1 - |complex cosine similarity|，
    使 OTF 的相位/符号信息通过实部、虚部共同进入距离计算。
    """
    Z = np.asarray(complex_representations)
    N = Z.shape[0]
    Z = Z.reshape(N, -1).astype(np.complex64)

    real_imag_features = np.concatenate(
        [Z.real, Z.imag],
        axis=1,
    ).astype(np.float32)

    return compute_RDM(
        real_imag_features,
        metric=metric,
        zscore=zscore,
    )


def upper_tri_vector(rdm):
    mask = np.triu(np.ones_like(rdm, dtype=bool), k=1)
    return np.asarray(rdm[mask], dtype=np.float64)


def rsa_correlation(rdm_a, rdm_b, method="pearson", eps=1e-12):
    """
    RSA = 两个 RDM 上三角向量之间的相关。
    method 可选：pearson / spearman
    """
    vec_a = upper_tri_vector(rdm_a)
    vec_b = upper_tri_vector(rdm_b)

    valid = np.isfinite(vec_a) & np.isfinite(vec_b)
    vec_a = vec_a[valid]
    vec_b = vec_b[valid]

    if len(vec_a) < 3:
        return np.nan, np.nan
    if np.std(vec_a) < eps or np.std(vec_b) < eps:
        return np.nan, np.nan

    if method == "pearson":
        r, p = pearsonr(vec_a, vec_b)
    elif method == "spearman":
        r, p = spearmanr(vec_a, vec_b)
    else:
        raise ValueError(f"Unsupported RSA method: {method}")

    return float(r), float(p)


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


def extract_layer_representations(net, a_rsa, device, forward_batch=512):
    h_lists = [[] for _ in range(len(net.layers))]
    N_rsa = a_rsa.shape[0]

    for start in range(0, N_rsa, forward_batch):
        end = min(start + forward_batch, N_rsa)
        batch_np = a_rsa[start:end]
        batch_t = torch.from_numpy(batch_np).float()

        acts = get_all_layer_activations(net, batch_t, device)

        for i in range(len(net.layers)):
            h_lists[i].append(acts[i].numpy())

    layer_repr_map = {}
    for i, chunks in enumerate(h_lists):
        layer_name = f"Layer{i + 1}"
        layer_repr_map[layer_name] = np.concatenate(chunks, axis=0).astype(np.float32)
        print(f"{layer_name}: {layer_repr_map[layer_name].shape}")

    return layer_repr_map


# ========================== 数据 / 模型加载 ==========================

def load_test_a(opt):
    train_iter, val_iter, test_iter = MyKANnetLoader_2.load_dataset(opt)

    a_list = []
    alpha_list = []

    for batch in test_iter:
        a, alpha = batch
        a_list.append(a)
        alpha_list.append(alpha)

    a_test = torch.cat(a_list, dim=0).numpy().reshape(-1, opt.input_dim)
    alpha_test = torch.cat(alpha_list, dim=0).numpy().reshape(-1, opt.output_dim)

    print("Full test set a_test:", a_test.shape)
    print("Full test set alpha_test:", alpha_test.shape)

    return a_test, alpha_test


def sample_for_rsa(a_test, max_rsa_samples=2000, seed=0):
    N_test = a_test.shape[0]

    if N_test > max_rsa_samples:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(N_test, size=max_rsa_samples, replace=False)
        a_rsa = a_test[sample_idx]
        print(f"Sample {max_rsa_samples} / {N_test} samples for group-vector RSA")
    else:
        sample_idx = np.arange(N_test)
        a_rsa = a_test
        print(f"Use all {N_test} samples for group-vector RSA")

    return a_rsa.astype(np.float32), sample_idx


def build_model(opt, device):
    layers_hidden = [opt.input_dim] + opt.hidden_dims + [opt.output_dim]
    print("layers_hidden:", layers_hidden)

    net = SymbolicKAN(
        layers_hidden=layers_hidden,
        elementary_functions=DEFAULT_ELEMENTARY_FUNCTIONS,
    ).to(device)

    checkpoint = torch.load(opt.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "parameter" in checkpoint:
        state_dict = checkpoint["parameter"]
    else:
        state_dict = checkpoint

    net.load_state_dict(state_dict)
    net.eval()

    return net


# ========================== 物理表征 / RDM ==========================

def compute_physical_rdm_map(a_rsa, zern_poly, grid_size=128, phys_metric="euclidean"):
    N_rsa = a_rsa.shape[0]
    phase_dim = grid_size * grid_size

    print("\n===== Calculating physical representations =====")

    phase_repr = np.zeros((N_rsa, phase_dim), dtype=np.float32)
    sin_phase_repr = np.zeros((N_rsa, phase_dim), dtype=np.float32)
    cos_phase_repr = np.zeros((N_rsa, phase_dim), dtype=np.float32)

    exp_phase_repr = np.zeros((N_rsa, grid_size, grid_size), dtype=np.complex64)
    wave_repr = np.zeros((N_rsa, grid_size, grid_size), dtype=np.complex64)
    otf_repr = np.zeros((N_rsa, grid_size, grid_size), dtype=np.complex64)

    for i in range(N_rsa):
        if (i + 1) % 200 == 0 or (i + 1) == N_rsa:
            print(f"  physical forward: {i + 1}/{N_rsa}")

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

    print("\n===== Computing physical RDMs =====")

    phys_rdm_map = {
        "Phase": compute_RDM(phase_repr, metric=phys_metric, zscore=False),
        "SinPhase": compute_RDM(sin_phase_repr, metric=phys_metric, zscore=False),
        "CosPhase": compute_RDM(cos_phase_repr, metric=phys_metric, zscore=False),
        "ExpPhase": compute_complex_RDM(exp_phase_repr),
        "Wavefront": compute_complex_RDM(wave_repr),
        # OTF 是复数表征：先拼接 [real(OTF), imag(OTF)]，再按实数特征矩阵计算 RDM。
        "OTF": compute_complex_realimag_concat_RDM(
            otf_repr,
            metric=phys_metric,
            zscore=False,
        ),
    }

    return phys_rdm_map


# ========================== 读取分类 CSV，并按层/类别聚合节点 ==========================

def load_classification_csv(classification_csv, class_col="best_phys", r_threshold=None):
    df = pd.read_csv(classification_csv)

    required_cols = {"layer", "neuron", class_col}
    missing = required_cols - set(df.columns)
    if len(missing) > 0:
        raise ValueError(
            f"classification_csv 缺少必要列: {missing}. "
            f"当前列为: {list(df.columns)}"
        )

    df = df.copy()
    df["neuron"] = df["neuron"].astype(int)
    df[class_col] = df[class_col].astype(str)

    if r_threshold is not None:
        if "best_r" not in df.columns:
            raise ValueError("设置了 --r_threshold，但 CSV 中没有 best_r 列。")
        before = len(df)
        df = df[df["best_r"] >= r_threshold].copy()
        print(f"Filter neurons by best_r >= {r_threshold}: {len(df)} / {before}")

    df = df.drop_duplicates(subset=["layer", "neuron"], keep="first")
    print("Loaded classification CSV:", classification_csv)
    print("Valid classified neurons:", len(df))

    return df


def build_group_vector_rsa(
    layer_repr_map,
    phys_rdm_map,
    class_df,
    class_col="best_phys",
    min_neurons_per_group=1,
    rsa_method="pearson",
    activation_metric="euclidean",
    zscore_group_features=True,
):
    """
    对每个 Layer 内同一类别的节点组成群体向量：
        group_repr = layer_repr[:, neuron_indices]
    然后计算 group_repr 的 RDM，并与各物理 RDM 做 RSA。
    """
    results = []
    members = []

    phys_names = list(phys_rdm_map.keys())

    for (layer_name, class_name), sub_df in class_df.groupby(["layer", class_col]):
        if layer_name not in layer_repr_map:
            print(f"[Skip] {layer_name} not found in extracted activations.")
            continue

        layer_repr = layer_repr_map[layer_name]
        hidden_dim = layer_repr.shape[1]

        neuron_indices = sorted(sub_df["neuron"].astype(int).unique().tolist())
        neuron_indices = [idx for idx in neuron_indices if 0 <= idx < hidden_dim]

        if len(neuron_indices) < min_neurons_per_group:
            print(
                f"[Skip] {layer_name} / {class_name}: "
                f"only {len(neuron_indices)} neurons < min_neurons_per_group"
            )
            continue

        if len(neuron_indices) == 0:
            continue

        group_repr = layer_repr[:, neuron_indices]

        # 若所有节点激活几乎常数，RDM 会退化。
        if np.std(group_repr) < 1e-12:
            print(f"[Skip] {layer_name} / {class_name}: constant group representation")
            continue

        group_rdm = compute_RDM(
            group_repr,
            metric=activation_metric,
            zscore=zscore_group_features,
        )

        row = {
            "layer": layer_name,
            "group_class": class_name,
            "n_neurons": len(neuron_indices),
            "neuron_indices": ",".join(map(str, neuron_indices)),
        }

        for phys_name in phys_names:
            r, p = rsa_correlation(group_rdm, phys_rdm_map[phys_name], method=rsa_method)
            row[f"RSA_{phys_name}"] = r
            row[f"p_{phys_name}"] = p
            row[f"Distance_1_minus_RSA_{phys_name}"] = np.nan if np.isnan(r) else 1.0 - r

        rsa_values = {phys_name: row[f"RSA_{phys_name}"] for phys_name in phys_names}
        finite_rsa_values = {k: v for k, v in rsa_values.items() if np.isfinite(v)}
        if len(finite_rsa_values) > 0:
            best_phys = max(finite_rsa_values.keys(), key=lambda k: finite_rsa_values[k])
            row["best_matched_phys_by_group_RSA"] = best_phys
            row["best_group_RSA"] = finite_rsa_values[best_phys]
            row["best_group_distance_1_minus_RSA"] = 1.0 - finite_rsa_values[best_phys]
        else:
            row["best_matched_phys_by_group_RSA"] = "NaN"
            row["best_group_RSA"] = np.nan
            row["best_group_distance_1_minus_RSA"] = np.nan

        results.append(row)
        members.append({
            "layer": layer_name,
            "group_class": class_name,
            "n_neurons": len(neuron_indices),
            "neuron_indices": ",".join(map(str, neuron_indices)),
        })

        print(
            f"{layer_name:>6s} | class={class_name:<10s} | "
            f"n={len(neuron_indices):4d} | "
            f"best={row['best_matched_phys_by_group_RSA']} | "
            f"RSA={row['best_group_RSA']:.4f} | "
            f"dist={row['best_group_distance_1_minus_RSA']:.4f}"
        )

    result_df = pd.DataFrame(results)
    member_df = pd.DataFrame(members)

    if len(result_df) > 0:
        result_df = result_df.sort_values(
            ["layer", "group_class"],
            ascending=[True, True],
        ).reset_index(drop=True)

    return result_df, member_df


# ========================== 保存矩阵与画图 ==========================

def make_matrix_tables(result_df, phys_names):
    row_names = [
        f"{row['layer']}__{row['group_class']}__n{int(row['n_neurons'])}"
        for _, row in result_df.iterrows()
    ]

    rsa_mat = pd.DataFrame(index=row_names, columns=phys_names, dtype=float)
    dist_mat = pd.DataFrame(index=row_names, columns=phys_names, dtype=float)

    for idx, row in result_df.iterrows():
        row_name = row_names[idx]
        for phys_name in phys_names:
            rsa_mat.loc[row_name, phys_name] = row[f"RSA_{phys_name}"]
            dist_mat.loc[row_name, phys_name] = row[f"Distance_1_minus_RSA_{phys_name}"]

    rsa_mat.index.name = "layer_group_vector"
    dist_mat.index.name = "layer_group_vector"

    return rsa_mat, dist_mat


def save_heatmap(matrix_df, save_path, title, cbar_label):
    if matrix_df.empty:
        print(f"[Skip] Empty matrix, no heatmap saved: {save_path}")
        return

    mat = matrix_df.astype(float).values

    fig_h = max(4, 0.35 * matrix_df.shape[0] + 1.5)
    fig_w = max(7, 0.8 * matrix_df.shape[1] + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto")

    ax.set_xticks(np.arange(matrix_df.shape[1]))
    ax.set_xticklabels(matrix_df.columns, rotation=35, ha="right")

    ax.set_yticks(np.arange(matrix_df.shape[0]))
    ax.set_yticklabels(matrix_df.index)

    ax.set_title(title)
    ax.set_xlabel("Physical representation")
    ax.set_ylabel("Layer / same-class neuron group vector")

    for i in range(matrix_df.shape[0]):
        for j in range(matrix_df.shape[1]):
            val = mat[i, j]
            text = "nan" if not np.isfinite(val) else f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmap: {save_path}")


def save_outputs(result_df, member_df, phys_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    result_csv = os.path.join(save_dir, "group_vector_rsa_results.csv")
    member_csv = os.path.join(save_dir, "group_vector_neuron_members.csv")
    rsa_matrix_csv = os.path.join(save_dir, "group_vector_rsa_matrix.csv")
    dist_matrix_csv = os.path.join(save_dir, "group_vector_distance_matrix.csv")

    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    member_df.to_csv(member_csv, index=False, encoding="utf-8-sig")

    rsa_mat, dist_mat = make_matrix_tables(result_df, phys_names)
    rsa_mat.to_csv(rsa_matrix_csv, encoding="utf-8-sig")
    dist_mat.to_csv(dist_matrix_csv, encoding="utf-8-sig")

    save_heatmap(
        dist_mat,
        os.path.join(save_dir, "group_vector_distance_heatmap.png"),
        title="Distance = 1 - RSA between group vectors and physical representations",
        cbar_label="1 - RSA",
    )

    save_heatmap(
        rsa_mat,
        os.path.join(save_dir, "group_vector_rsa_heatmap.png"),
        title="RSA between group vectors and physical representations",
        cbar_label="RSA",
    )

    meta = {
        "result_csv": result_csv,
        "member_csv": member_csv,
        "rsa_matrix_csv": rsa_matrix_csv,
        "dist_matrix_csv": dist_matrix_csv,
    }
    with open(os.path.join(save_dir, "output_files.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n===== Saved outputs =====")
    for k, v in meta.items():
        print(f"{k}: {v}")


# ========================== 主程序 ==========================

def parse_hidden_dims(s):
    if isinstance(s, list):
        return s
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def main():
    parser = argparse.ArgumentParser()

    # 与原始数据加载保持兼容
    parser.add_argument(
        "--data_root",
        type=str,
        default="/media/aiofm/F/20250723_k-fold-cross-validation-KAN/1-fold",
    )
    parser.add_argument("--batchSize", type=int, default=5120)
    parser.add_argument(
        "--pca_model_path",
        type=str,
        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/New_pca_model-70.pkl",
    )

    # 模型 / 物理表征路径
    parser.add_argument("--zernike_poly_path", type=str, default="36—128ZernPoly.npy")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/Last_SymbolicKAN_Para_Silu_1_fold_4L_-524-524-524_15_2.pt",
    )

    # 读取前一步生成的神经元分类 CSV
    parser.add_argument(
        "--classification_csv",
        type=str,
        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_RSA/neuron_level_rsa/neuron_level_rsa_results.csv",
        help="前一步 neuron-level RSA 生成的 CSV，例如 neuron_level_rsa_results.csv 或 neuron_level_rsa_results_r_ge_0.1.csv",
    )

    # 输出
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_RSA/group_vector_rsa",
    )

    # 网络结构
    parser.add_argument("--input_dim", type=int, default=33)
    parser.add_argument("--hidden_dims", type=str, default="524,524,524")
    parser.add_argument("--output_dim", type=int, default=70)

    # RSA 参数
    parser.add_argument("--max_rsa_samples", type=int, default=2000)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--forward_batch", type=int, default=512)
    parser.add_argument("--grid_size", type=int, default=128)
    parser.add_argument("--rsa_method", type=str, default="pearson", choices=["pearson", "spearman"])
    parser.add_argument("--activation_metric", type=str, default="euclidean")
    parser.add_argument("--phys_metric", type=str, default="euclidean")

    # 分组参数
    parser.add_argument("--class_col", type=str, default="best_phys")
    parser.add_argument("--r_threshold", type=float, default=None)
    parser.add_argument("--min_neurons_per_group", type=int, default=1)
    parser.add_argument(
        "--no_zscore_group_features",
        action="store_true",
        help="默认对群体节点向量按特征维度 z-score；加此参数则不做 z-score。",
    )

    opt = parser.parse_args()
    opt.hidden_dims = parse_hidden_dims(opt.hidden_dims)

    os.makedirs(opt.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. 读取分类 CSV
    class_df = load_classification_csv(
        classification_csv=opt.classification_csv,
        class_col=opt.class_col,
        r_threshold=opt.r_threshold,
    )

    # 2. 加载测试集并采样，保证网络激活 RDM 与物理 RDM 使用同一批样本
    a_test, alpha_test = load_test_a(opt)
    a_rsa, sample_idx = sample_for_rsa(
        a_test,
        max_rsa_samples=opt.max_rsa_samples,
        seed=opt.sample_seed,
    )
    np.save(os.path.join(opt.save_dir, "rsa_sample_indices.npy"), sample_idx)

    # 3. 计算物理 RDM
    zern_poly = load_zernike_poly(opt.zernike_poly_path)
    phys_rdm_map = compute_physical_rdm_map(
        a_rsa,
        zern_poly=zern_poly,
        grid_size=opt.grid_size,
        phys_metric=opt.phys_metric,
    )
    phys_names = list(phys_rdm_map.keys())

    # 4. 加载模型并提取每一层激活
    print("\n===== Loading model and extracting layer activations =====")
    net = build_model(opt, device)
    layer_repr_map = extract_layer_representations(
        net=net,
        a_rsa=a_rsa,
        device=device,
        forward_batch=opt.forward_batch,
    )

    # 5. 每层内同类节点组成群体向量，与各物理量做 RSA，距离 = 1 - RSA
    print("\n===== Running group-vector RSA =====")
    result_df, member_df = build_group_vector_rsa(
        layer_repr_map=layer_repr_map,
        phys_rdm_map=phys_rdm_map,
        class_df=class_df,
        class_col=opt.class_col,
        min_neurons_per_group=opt.min_neurons_per_group,
        rsa_method=opt.rsa_method,
        activation_metric=opt.activation_metric,
        zscore_group_features=(not opt.no_zscore_group_features),
    )

    # 6. 保存结果
    save_outputs(
        result_df=result_df,
        member_df=member_df,
        phys_names=phys_names,
        save_dir=opt.save_dir,
    )

    print("\n===== Done =====")


if __name__ == "__main__":
    main()
