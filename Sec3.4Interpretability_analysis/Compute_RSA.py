import os
import math
import matplotlib.pyplot as plt
import joblib
import numpy as np
import sympy as sp
from scipy.signal import correlate
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import efficient_kan.kan as E_kan
import MyKANnetLoader_2
import argparse
# 新增热力图依赖
import matplotlib as mpl

# 关键：设置无GUI绘图后端，仅保存图片不弹出窗口
mpl.use('Agg')

# ========================== 全局配置初等函数 ==========================
SUPPORTED_ELEMENTARY_FUNCTIONS = {
    'silu': lambda x: torch.nn.functional.silu(x),
    'relu': lambda x: torch.nn.functional.relu(x),
    'sin': lambda x: torch.sin(x),
    'cos': lambda x: torch.cos(x),
    'exp': lambda x: torch.exp(torch.clamp(x, -10, 10)),
    'log': lambda x: torch.log(torch.abs(x) + 1e-6),
    'tanh': lambda x: torch.tanh(x),
    'sigmoid': lambda x: torch.sigmoid(x),
    'sqrt': lambda x: torch.sqrt(torch.abs(x) + 1e-6),
    'square': lambda x: torch.square(x),
    'abs': lambda x: torch.abs(x),
    'identity': lambda x: x
}

DEFAULT_ELEMENTARY_FUNCTIONS = ['silu', 'relu', 'tanh', 'sigmoid', 'abs', 'identity']
# DEFAULT_ELEMENTARY_FUNCTIONS = ['identity','square','sin','cos','abs','tanh']

# ========================== SymbolicKAN 网络定义 ==========================
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
            invalid_funcs = [f for f in elementary_functions if f not in SUPPORTED_ELEMENTARY_FUNCTIONS]
            if invalid_funcs:
                raise ValueError(
                    f"不支持的初等函数: {invalid_funcs}，支持的函数列表: {list(SUPPORTED_ELEMENTARY_FUNCTIONS.keys())}")
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
        mlp_output = nn.functional.linear(x, self.ef_mlp_linears[func_idx], self.ef_mlp_biases[func_idx])
        ef_output = SUPPORTED_ELEMENTARY_FUNCTIONS[func_name](mlp_output)
        return ef_output * self.ef_weights[func_idx]

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = nn.functional.linear(self.base_activation(x), self.base_weight)
        mlp_outputs = []
        for i in range(self.num_ef):
            ef_mlp_output = self.apply_elementary_function(x, i)
            mlp_outputs.append(ef_mlp_output)
        mlp_output = torch.stack(mlp_outputs, dim=-1).sum(dim=-1) * self.scale_mlp
        output = base_output + mlp_output
        return output.reshape(*original_shape[:-1], self.out_features)

    def get_l1_regularization(self):
        return self.ef_weights.abs().sum()


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
            invalid_funcs = [f for f in elementary_functions if f not in SUPPORTED_ELEMENTARY_FUNCTIONS]
            if invalid_funcs:
                raise ValueError(
                    f"不支持的初等函数: {invalid_funcs}，支持的函数列表: {list(SUPPORTED_ELEMENTARY_FUNCTIONS.keys())}")
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

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_l1_regularization(self):
        total_l1 = 0.0
        for layer in self.layers:
            total_l1 += layer.get_l1_regularization()
        return total_l1

# ========================== 物理流程拆分5个子层 a→phase→exp_phase→wave→otf→alpha ==========================
def func_a2phase(a, zern_poly):
    """a(33,) -> phase(128,128)"""
    phase = np.sum(a[:, None, None] * zern_poly, axis=0)
    return phase

def func_phase2exp(phase):
    """phase(128,128) -> exp(1j*2π phase) 复指数层"""
    exp_phase = np.exp(1j * 2 * np.pi * phase)
    return exp_phase

def func_exp2wave(exp_phase, N=128):
    """复指数项 + 光瞳mask -> complex wave(128,128)"""
    x, y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    mask = (np.sqrt(x ** 2 + y ** 2) <= 1).astype(float)
    wave = mask * exp_phase
    # 能量归一化
    p = np.sum(np.abs(wave) ** 2)
    wave = wave * (((1 / (N ** 2)) / p) ** 0.5)
    return wave

def func_wave2otf(wave, N=128):
    """complex wave -> OTF"""
    otf = correlate(wave, wave, mode="same") * (N ** 2)
    otf = otf[::-1, ::-1]
    return otf

def func_otf2alpha(otf, pca_model):
    """OTF -> PCA降维alpha"""
    N = otf.shape[0]
    flat = np.hstack((otf.real.ravel(), otf.imag.ravel())).reshape(1, 2 * N * N)
    alpha = pca_model.transform(flat).ravel()
    return alpha

def physical_forward(a, zern_poly, pca_model):
    """完整物理前向，返回新增复指数层exp_phase"""
    phase = func_a2phase(a, zern_poly)
    exp_phase = func_phase2exp(phase)
    wave = func_exp2wave(exp_phase)
    otf = func_wave2otf(wave)
    alpha = func_otf2alpha(otf, pca_model)
    return phase, exp_phase, wave, otf, alpha

# ========================== 原有物理计算函数 ==========================
def compute_kappa(Cn2=15e-16, wvl=0.525e-6, L=7000, D=0.305):
    delta0 = L * wvl / (2 * D)
    z = sp.symbols("z")
    expression = (z / L) ** (5 / 3)
    r0 = ((0.423 * (2 * np.pi / wvl) ** 2) * Cn2 * sp.integrate(expression, (z, 0, L))) ** (-3 / 5)
    kappa = (
        (D / r0) ** (5 / 3)
        / (2 ** (5 / 3))
        * (2 * wvl / (np.pi * D)) ** 2
        * 2 * np.pi
    ) ** 0.5 * L / delta0
    return float(kappa)

def generate_a(kappa, num_zern=36, seed=None):
    if seed is not None:
        np.random.seed(seed)
    C = np.eye(num_zern)
    b = np.random.randn(num_zern, 1)
    a = np.squeeze(C @ b)[3:]
    return a * kappa

def load_zernike_poly(path):
    z = np.load(path)
    z = np.transpose(z, (2, 0, 1))
    return z[3:, :, :].astype(np.float64)

def visualize_otf(otf, title="OTF"):
    amp = np.abs(otf)
    phase = np.angle(otf)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(amp, cmap='viridis')
    plt.title(title + " - Magnitude")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(phase, cmap='twilight')
    plt.title(title + " - Phase")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300, bbox_inches="tight")
    plt.close()

def compute_otf_from_a(a, zern_poly, N=128):
    x, y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    mask = np.sqrt(x ** 2 + y ** 2) <= 1
    mask = mask.astype(int)
    phase = np.sum(a[:, None, None] * zern_poly, axis=0)
    exp_phase = np.exp(1j * 2 * np.pi * phase)
    wave = mask * exp_phase
    p = np.sum(np.abs(wave) ** 2)
    wave = wave * (((1 / 128 ** 2) / p) ** 0.5)
    otf = correlate(wave, wave, mode="same") * N ** 2
    return otf[::-1, ::-1]

def compute_alpha_from_a(a, zern_poly, pca_model):
    otf = compute_otf_from_a(a, zern_poly)
    matrix_size = otf.shape[1]
    real_matrix = np.hstack((otf.real.ravel(), otf.imag.ravel())).reshape(1, 2 * matrix_size * matrix_size)
    alpha = pca_model.transform(real_matrix).ravel()
    return otf, alpha

def compute_kan_output(a, net, device):
    net.eval()
    a_tensor = torch.tensor(a, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(a_tensor).cpu().numpy().squeeze()
    return out

def select_taylor_point_full_test(a_test):
    mean = np.mean(a_test, axis=0)
    dist = np.linalg.norm(a_test - mean, axis=1)
    idx = np.argmin(dist)
    return a_test[idx], idx

# ========================== RSA 相关工具函数 ==========================
def get_all_layer_activations(net, x_batch, device):
    """前向钩子提取SymbolicKAN全部4层输出"""
    activations = []
    def hook_fn(module, inp, out):
        activations.append(out.detach().cpu())
    hooks = []
    for layer in net.layers:
        h = layer.register_forward_hook(hook_fn)
        hooks.append(h)
    net.eval()
    with torch.no_grad():
        _ = net(x_batch.to(device))
    for h in hooks:
        h.remove()
    return activations

def compute_RDM(representations):
    """输入[N, dim]表征，返回[N,N]欧氏距离RDM矩阵"""
    dist_vec = pdist(representations, metric='euclidean')
    rdm = squareform(dist_vec)
    return rdm

def rsa_correlation(rdm_phys, rdm_net):
    """两个RDM上三角扁平化计算皮尔逊相关"""
    mask = np.triu(np.ones_like(rdm_phys, dtype=bool), k=1)
    vec_phys = rdm_phys[mask]
    vec_net = rdm_net[mask]
    r, p = pearsonr(vec_phys, vec_net)
    return r, p

# ===================== 绘制RSA相关系数热力图（纯保存，无弹窗） =====================
def plot_rsa_heatmap(all_rsa_results, save_path="rsa_correlation_heatmap.png"):
    # 固定顺序
    phys_names = [
        "Physical Phase Field",
        "Physical Exp(1j*2π Phase)",
        "Physical Wavefront",
        "Physical OTF"
    ]
    net_names = [
        "Layer1 (524 dim)",
        "Layer2 (524 dim)",
        "Layer3 (524 dim)",
        "Layer4 Pred Alpha (70 dim)"
    ]
    # 组装r值矩阵
    r_mat = np.zeros((len(phys_names), len(net_names)))
    for i, p_name in enumerate(phys_names):
        layer_dict = all_rsa_results[p_name]
        for j, n_name in enumerate(net_names):
            r_mat[i, j] = layer_dict[n_name]["r"]

    # 绘图
    plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(9, 5))
    # 热力图，相关系数范围[-1,1]
    im = ax.imshow(r_mat, cmap="coolwarm", vmin=-1, vmax=1)

    # 坐标轴标签
    ax.set_xticks(np.arange(len(net_names)))
    ax.set_yticks(np.arange(len(phys_names)))
    ax.set_xticklabels([s.split(" ")[0] for s in net_names], fontsize=10)
    ax.set_yticklabels([
        "Phase Field",
        "Exp(2πj·Phase)",
        "Wavefront",
        "OTF"
    ], fontsize=10)
    ax.set_xlabel("Network Layers", fontsize=12)
    ax.set_ylabel("Physical Representations", fontsize=12)
    ax.set_title("RSA Pearson Correlation Coefficient Heatmap", fontsize=14, pad=15)

    # 每个格子标注r值，保留4位小数
    for i in range(len(phys_names)):
        for j in range(len(net_names)):
            text = ax.text(j, i, f"{r_mat[i, j]:.4f}",
                           ha="center", va="center", color="black", fontsize=9)

    # 色条
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson Correlation r", rotation=270, labelpad=15)

    plt.tight_layout()
    # 保存高清png
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # 保存矢量svg
    svg_path = save_path.replace(".png", ".svg")
    plt.savefig(svg_path, bbox_inches="tight")
    # 关闭画布释放内存，删除弹窗代码plt.show()
    plt.close(fig)
    print(f"\n热力图保存完成：")
    print(f"PNG文件：{save_path}")
    print(f"SVG矢量图：{svg_path}")

# ========================== 主程序入口 ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="/media/aiofm/F/20250723_k-fold-cross-validation-KAN/1-fold")
parser.add_argument('--batchSize', type=int, default=5120)
opt = parser.parse_args()

if __name__ == "__main__":
    # 路径配置
    zernike_poly_path = "36—128ZernPoly.npy"
    pca_model_path = "/home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/New_pca_model-70.pkl"
    model_path = "/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/Last_SymbolicKAN_Para_Silu_1_fold_4L_-524-524-524_15_2.pt"
    # model_path = "/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/Last_SymbolicKAN_Para_Cos_1_fold_4L_-524-524-524_15_2.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型、PCA、Zernike基
    pca = joblib.load(pca_model_path)
    zern_poly = load_zernike_poly(zernike_poly_path)
    layers_hidden = [33, 524, 524, 524, 70]
    net = SymbolicKAN(layers_hidden=layers_hidden, elementary_functions=DEFAULT_ELEMENTARY_FUNCTIONS).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint['parameter'])

    # 加载测试集
    train_iter, val_iter, test_iter = MyKANnetLoader_2.load_dataset(opt)
    a_list, alpha_list = [], []
    for batch in test_iter:
        a, alpha = batch
        a_list.append(a)
        alpha_list.append(alpha)
    a_test = torch.cat(a_list, dim=0).numpy().reshape(-1, 33)
    alpha_test = torch.cat(alpha_list, dim=0).numpy().reshape(-1, 70)
    N_test = a_test.shape[0]
    print("Full test set shape a_test:", a_test.shape)

    # ====================== RSA 分析开始 ======================
    MAX_RSA_SAMPLES = 2000
    if N_test > MAX_RSA_SAMPLES:
        sample_idx = np.random.choice(N_test, size=MAX_RSA_SAMPLES, replace=False)
        a_rsa = a_test[sample_idx]
        print(f"\nTest set too large ({N_test}), random sample {MAX_RSA_SAMPLES} for RSA")
    else:
        a_rsa = a_test
    N_rsa = a_rsa.shape[0]
    grid_size = 128
    phase_dim = grid_size * grid_size
    complex_dim = 2 * grid_size * grid_size

    # 1. 批量计算四类物理表征 Phase / ExpPhase(新增) / Wave / OTF
    print("\n===== Calculating all physical representations (Phase + ExpPhase + Wave + OTF) =====")
    phase_repr = np.zeros((N_rsa, phase_dim), dtype=np.float32)
    exp_phase_repr = np.zeros((N_rsa, complex_dim), dtype=np.float32)  # 新增复指数层表征
    wave_repr = np.zeros((N_rsa, complex_dim), dtype=np.float32)
    otf_repr = np.zeros((N_rsa, complex_dim), dtype=np.float32)

    for i in range(N_rsa):
        a_samp = a_rsa[i]
        # 分层分步计算
        phase = func_a2phase(a_samp, zern_poly)
        exp_phase = func_phase2exp(phase)
        wave = func_exp2wave(exp_phase)
        otf = func_wave2otf(wave)

        # Phase：直接展平实数
        phase_repr[i] = phase.ravel()
        # ExpPhase：复数值拆实虚拼接
        exp_flat = np.hstack([exp_phase.real.ravel(), exp_phase.imag.ravel()])
        exp_phase_repr[i] = exp_flat
        # Wave：实部虚部拼接
        wave_flat = np.hstack([wave.real.ravel(), wave.imag.ravel()])
        wave_repr[i] = wave_flat
        # OTF：实部虚部拼接
        otf_flat = np.hstack([otf.real.ravel(), otf.imag.ravel()])
        otf_repr[i] = otf_flat

    print(f"phase_repr shape: {phase_repr.shape}")
    print(f"exp_phase_repr shape: {exp_phase_repr.shape}")
    print(f"wave_repr shape: {wave_repr.shape}")
    print(f"otf_repr shape: {otf_repr.shape}")

    # 2. 分批前向提取KAN四层激活
    print("\n===== Extracting SymbolicKAN layer activations =====")
    batch_r = 512
    h1_list, h2_list, h3_list, h4_list = [], [], [], []
    for start in range(0, N_rsa, batch_r):
        end = min(start + batch_r, N_rsa)
        batch_np = a_rsa[start:end]
        batch_t = torch.from_numpy(batch_np).float()
        acts = get_all_layer_activations(net, batch_t, device)
        h1, h2, h3, h4 = acts
        h1_list.append(h1.numpy())
        h2_list.append(h2.numpy())
        h3_list.append(h3.numpy())
        h4_list.append(h4.numpy())
    h1_repr = np.concatenate(h1_list, axis=0)
    h2_repr = np.concatenate(h2_list, axis=0)
    h3_repr = np.concatenate(h3_list, axis=0)
    h4_repr = np.concatenate(h4_list, axis=0)

    print(f"Layer1 shape: {h1_repr.shape}, Layer2: {h2_repr.shape}, Layer3: {h3_repr.shape}, Layer4(output): {h4_repr.shape}")

    # 表征映射：新增复指数层 Exp(1j2π phase)
    net_repr_map = {
        "Layer1 (524 dim)": h1_repr,
        "Layer2 (524 dim)": h2_repr,
        "Layer3 (524 dim)": h3_repr,
        "Layer4 Pred Alpha (70 dim)": h4_repr
    }
    phys_repr_map = {
        "Physical Phase Field": phase_repr,
        "Physical Exp(1j*2π Phase)": exp_phase_repr,  # 新增独立一层
        "Physical Wavefront": wave_repr,
        "Physical OTF": otf_repr
    }

    # 3. 批量计算所有配对 RSA 相关系数
    all_rsa_results = {}
    print("\n==================== RSA Correlation Results ====================")
    for phys_name, phys_mat in phys_repr_map.items():
        print(f"\n----------- Benchmark: {phys_name} -----------")
        rdm_phys = compute_RDM(phys_mat)
        layer_res = {}
        for net_name, net_mat in net_repr_map.items():
            rdm_net = compute_RDM(net_mat)
            r, p = rsa_correlation(rdm_phys, rdm_net)
            layer_res[net_name] = {"r": r, "pval": p}
            print(f"{net_name}: r = {r:.4f}, p-value = {p:.2e}")
        all_rsa_results[phys_name] = layer_res

    # ===================== 绘制并保存热力图 =====================
    plot_rsa_heatmap(all_rsa_results, save_path="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_RSA/rsa_heatmap_result.png")
