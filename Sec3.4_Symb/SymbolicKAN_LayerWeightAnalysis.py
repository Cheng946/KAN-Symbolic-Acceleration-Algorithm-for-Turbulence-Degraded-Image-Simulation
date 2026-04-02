import torch
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm  # 进度条库
import time  # 计时

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 原有SymbolicKAN定义（无修改） ==========
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


class SymbolicKANLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, scale_base=1.0, scale_mlp=1.0,
                 base_activation=torch.nn.SiLU, elementary_functions=None):
        super().__init__()
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
                raise ValueError(f"不支持的初等函数: {invalid_funcs}")
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
        import math
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        for i in range(self.num_ef):
            torch.nn.init.kaiming_uniform_(self.ef_mlp_linears[i], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.ef_mlp_linears[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.ef_mlp_biases[i], -bound, bound)
        torch.nn.init.constant_(self.ef_weights, 1.0 / self.num_ef)

    def apply_elementary_function(self, x, func_idx):
        import torch.nn.functional as F
        func_name = self.elementary_functions[func_idx]
        mlp_output = F.linear(x, self.ef_mlp_linears[func_idx], self.ef_mlp_biases[func_idx])
        ef_output = SUPPORTED_ELEMENTARY_FUNCTIONS[func_name](mlp_output)
        return ef_output * self.ef_weights[func_idx]

    def forward(self, x):
        import torch.nn.functional as F
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
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
    def __init__(self, layers_hidden, scale_base=1.0, scale_mlp=1.0,
                 base_activation=torch.nn.SiLU, elementary_functions=None):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if elementary_functions is None:
            ef_list = DEFAULT_ELEMENTARY_FUNCTIONS
        else:
            invalid_funcs = [f for f in elementary_functions if f not in SUPPORTED_ELEMENTARY_FUNCTIONS]
            if invalid_funcs:
                raise ValueError(f"不支持的初等函数: {invalid_funcs}")
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

    def get_l1_regularization(self):
        total_l1 = 0.0
        for layer in self.layers:
            total_l1 += layer.get_l1_regularization()
        return total_l1


# ========== 全局变量用于记录第一幅图和最后一幅图的统计信息 ==========
first_plot_stats = {}
last_plot_stats = {}
all_plot_stats = []  # 存储所有图的统计信息


# ========== 核心优化：突出重点的热力图绘制函数（添加进度提示和统计功能） ==========
def plot_ef_weight_heatmap(weight_matrix, layer_idx, func_name, save_dir,
                           in_features, out_features, figsize=(12, 8),
                           highlight_percentile=95, mask_small_values=True,
                           small_value_threshold=0.01, is_first_plot=False, is_last_plot=False):
    """
    绘制突出重点的权重热力图，并添加统计功能
    参数：
        weight_matrix: 权重矩阵 (out_features, in_features)
        layer_idx: 层索引
        func_name: 初等函数名称
        save_dir: 保存目录
        in_features/out_features: 输入/输出维度
        highlight_percentile: 高亮显示的分位数（95表示只显示前5%的大权重）
        mask_small_values: 是否屏蔽小值（设为0）
        small_value_threshold: 小值阈值（绝对值小于此值设为0）
        is_first_plot: 是否是第一幅图
        is_last_plot: 是否是最后一幅图
    """
    global first_plot_stats, last_plot_stats

    print(f"\n[进度] 开始绘制 Layer {layer_idx} - {func_name} 热力图...")
    start_time = time.time()

    # 1. 数据预处理：屏蔽小值，突出大权重
    weight_matrix_processed = weight_matrix.copy()

    # 方法1：屏蔽极小值（减少噪声）
    if mask_small_values:
        weight_matrix_processed[np.abs(weight_matrix_processed) < small_value_threshold] = 0

    # 方法2：计算分位数，只高亮重要权重
    abs_weights = np.abs(weight_matrix_processed)
    highlight_threshold = np.percentile(abs_weights[abs_weights > 0], highlight_percentile) if np.any(
        abs_weights > 0) else 0

    # ========== 新增统计功能 ==========
    # 1. 计算95%分布的横坐标范围（权重值的2.5%和97.5%分位数）
    flattened_weights = weight_matrix_processed.flatten()
    valid_weights = flattened_weights[flattened_weights != 0]  # 排除屏蔽的小值
    if len(valid_weights) > 0:
        q2_5 = np.percentile(valid_weights, 2.5)
        q97_5 = np.percentile(valid_weights, 97.5)
        weight_range_95 = (q2_5, q97_5)
    else:
        weight_range_95 = (0, 0)

    # 2. 统计中高权重分布最多的维度
    # 中高权重定义：大于等于highlight_threshold的权重
    high_weight_mask = abs_weights >= highlight_threshold

    # 统计每列（输入维度）的高权重数量
    col_high_weight_counts = high_weight_mask.sum(axis=0)  # (in_features,)
    # 统计每行（输出维度）的高权重数量
    row_high_weight_counts = high_weight_mask.sum(axis=1)  # (out_features,)

    # 获取中高权重分布最多的5列维度序号（降序）
    top_5_cols = np.argsort(col_high_weight_counts)[-5:][::-1]
    top_5_cols_counts = col_high_weight_counts[top_5_cols]

    # 获取高权重分布最多的5行维度序号（降序）
    top_5_rows = np.argsort(row_high_weight_counts)[-5:][::-1]
    top_5_rows_counts = row_high_weight_counts[top_5_rows]

    # 3. 记录统计信息
    plot_stats = {
        'layer_idx': layer_idx,
        'func_name': func_name,
        'weight_range_95': weight_range_95,
        'top_5_cols': top_5_cols,
        'top_5_cols_counts': top_5_cols_counts,
        'top_5_rows': top_5_rows,
        'top_5_rows_counts': top_5_rows_counts,
        'highlight_threshold': highlight_threshold
    }

    all_plot_stats.append(plot_stats)

    # 记录第一幅图和最后一幅图的统计信息
    if is_first_plot:
        first_plot_stats = plot_stats
        print(f"\n[第一幅图统计] Layer {layer_idx} - {func_name}")
        print(f"  95%权重分布横坐标范围: [{weight_range_95[0]:.6f}, {weight_range_95[1]:.6f}]")
        print(f"  中高权重分布最多的5列维度序号: {top_5_cols}")
        print(f"  对应列的高权重数量: {top_5_cols_counts}")

    if is_last_plot:
        last_plot_stats = plot_stats
        print(f"\n[最后一幅图统计] Layer {layer_idx} - {func_name}")
        print(f"  95%权重分布横坐标范围: [{weight_range_95[0]:.6f}, {weight_range_95[1]:.6f}]")
        print(f"  高权重分布最多的5行维度序号: {top_5_rows}")
        print(f"  对应行的高权重数量: {top_5_rows_counts}")

    # 打印当前图的95%分布范围
    print(f"[统计] {func_name} 权重95%分布横坐标范围: [{weight_range_95[0]:.6f}, {weight_range_95[1]:.6f}]")

    # 2. 创建图形
    fig, (ax_main, ax_hist) = plt.subplots(2, 1, figsize=figsize,
                                           gridspec_kw={'height_ratios': [4, 1]})

    # 3. 绘制主热力图（蓝红配色）
    # 设置颜色范围为分位数范围，增强对比
    vmin = -highlight_threshold if np.min(weight_matrix_processed) < 0 else 0
    vmax = highlight_threshold if np.max(weight_matrix_processed) > 0 else 0

    im = ax_main.pcolor(weight_matrix_processed, cmap='coolwarm', vmin=vmin, vmax=vmax)

    # 4. 标注重点权重位置（前5%的大权重）
    y_coords, x_coords = np.where(abs_weights >= highlight_threshold)
    for x, y in zip(x_coords, y_coords):
        ax_main.scatter(x + 0.5, y + 0.5, s=10, c='yellow', marker='*',
                        edgecolors='black', linewidths=0.5, zorder=5,
                        label='Top 5% Weights' if (x == x_coords[0] and y == y_coords[0]) else "")

    # 5. 优化主图样式
    ax_main.set_xlabel(f'Input Dimension (total: {in_features})', fontsize=12, fontweight='bold')
    ax_main.set_ylabel(f'Output Dimension (total: {out_features})', fontsize=12, fontweight='bold')
    ax_main.set_title(
        f'Layer {layer_idx} - Top Function: {func_name}\nHighlighted Important Weights (Top {100 - highlight_percentile}%)',
        fontsize=14, fontweight='bold', pad=20)

    # 简化刻度（大维度只显示关键刻度）
    if in_features > 50:
        ax_main.set_xticks([0, in_features // 4, in_features // 2, 3 * in_features // 4, in_features - 1])
        ax_main.set_xticklabels([0, in_features // 4, in_features // 2, 3 * in_features // 4, in_features - 1])
    if out_features > 50:
        ax_main.set_yticks([0, out_features // 4, out_features // 2, 3 * out_features // 4, out_features - 1])
        ax_main.set_yticklabels([0, out_features // 4, out_features // 2, 3 * out_features // 4, out_features - 1])

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax_main)
    cbar.set_label('Weight Value (Small Values Masked)', fontsize=10, fontweight='bold')

    # 添加图例（只显示一次）
    handles, labels = ax_main.get_legend_handles_labels()
    if handles:
        ax_main.legend(handles, labels, loc='upper right', fontsize=8)

    # 6. 添加权重分布直方图（辅助分析）
    ax_hist.hist(weight_matrix_processed.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax_hist.axvline(highlight_threshold, color='red', linestyle='--', linewidth=1,
                    label=f'{highlight_percentile}th Percentile')
    ax_hist.axvline(-highlight_threshold, color='red', linestyle='--', linewidth=1)
    # 添加95%分布范围的标注
    ax_hist.axvline(weight_range_95[0], color='green', linestyle=':', linewidth=1.5,
                    label='2.5th Percentile (95% range)')
    ax_hist.axvline(weight_range_95[1], color='green', linestyle=':', linewidth=1.5,
                    label='97.5th Percentile (95% range)')
    ax_hist.set_xlabel('Weight Value', fontsize=10)
    ax_hist.set_ylabel('Count', fontsize=10)
    ax_hist.set_title('Weight Distribution (Masked)', fontsize=10, fontweight='bold')
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.3)

    # 7. 整体布局优化
    plt.tight_layout()

    # 8. 保存图片（高清）
    img_filename = f"layer_{layer_idx}_top_func_{func_name}_weights_highlighted.png"
    img_path = os.path.join(save_dir, img_filename)
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    elapsed_time = time.time() - start_time
    print(f"[进度] 热力图绘制完成 (耗时: {elapsed_time:.2f}秒)")
    print(f"[保存] 突出重点的热力图已保存至: {img_path}")
    print(f"  - 屏蔽了绝对值 < {small_value_threshold} 的小权重")
    print(f"  - 高亮显示了前 {100 - highlight_percentile}% 的重要权重（黄色星号）")
    print(f"  - 95%权重分布范围: [{weight_range_95[0]:.6f}, {weight_range_95[1]:.6f}]")

    return plot_stats


# ========== 主提取函数（添加完整进度提示和统计输出） ==========
def extract_ef_weights_per_layer(model_path, layers_hidden, elementary_functions,
                                 save_dir="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_SymbolicParam",
                                 device="cuda", plot_heatmaps=True):
    # 初始化全局统计变量
    global first_plot_stats, last_plot_stats, all_plot_stats
    first_plot_stats = {}
    last_plot_stats = {}
    all_plot_stats = []

    # 初始化总进度
    total_steps = len(layers_hidden) - 1  # 总层数
    current_step = 0

    print("=" * 60)
    print("开始执行 SymbolicKAN 权重提取流程")
    print("=" * 60)
    print(f"[配置信息]")
    print(f"  - 模型路径: {model_path}")
    print(f"  - 网络结构: {layers_hidden} (共{total_steps}层)")
    print(f"  - 初等函数: {elementary_functions}")
    print(f"  - 保存目录: {save_dir}")
    print(f"  - 设备: {device}")
    print(f"  - 是否绘制热力图: {plot_heatmaps}")
    print("=" * 60)

    # 步骤1: 创建保存目录
    print(f"\n[步骤 {current_step + 1}/{total_steps + 2}] 创建保存目录...")
    start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)
    elapsed_time = time.time() - start_time
    print(f"[完成] 保存目录准备就绪 (耗时: {elapsed_time:.2f}秒)")
    current_step += 1

    # 步骤2: 加载模型
    print(f"\n[步骤 {current_step + 1}/{total_steps + 2}] 加载模型...")
    start_time = time.time()
    model = SymbolicKAN(
        layers_hidden=layers_hidden,
        elementary_functions=elementary_functions
    ).to(device)

    # 加载权重
    print(f"  - 正在读取模型权重文件...")
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        print(f"  - 已移除 state_dict 中的 'module.' 前缀")

    if 'parameter' in state_dict:
        model.load_state_dict(state_dict['parameter'])
        print(f"  - 从 'parameter' 键加载权重")
    else:
        model.load_state_dict(state_dict)

    elapsed_time = time.time() - start_time
    print(f"[完成] 模型加载完成 (耗时: {elapsed_time:.2f}秒)")
    current_step += 1

    # 步骤3: 逐层提取权重
    ef_weights_dict = {}
    model.eval()
    print(f"\n[步骤 {current_step + 1}/{total_steps + 2}] 逐层提取权重 (共{total_steps}层)")
    print("-" * 50)

    with torch.no_grad():
        # 使用进度条显示逐层处理进度
        for layer_idx, layer in enumerate(tqdm(model.layers, desc="处理各层", unit="层")):
            layer_start_time = time.time()
            current_layer_step = layer_idx + 1

            # 判断是否是第一层/最后一层（用于统计）
            is_first = (layer_idx == 0) and plot_heatmaps
            is_last = (layer_idx == len(model.layers) - 1) and plot_heatmaps

            tqdm.write(f"\n[层 {current_layer_step}/{total_steps}] 处理 Layer {layer_idx}")
            tqdm.write(f"  - 输入维度: {layer.in_features}, 输出维度: {layer.out_features}")

            # 提取权重
            ef_weights = layer.ef_weights.cpu().numpy()
            ef_weights_normalized = ef_weights / ef_weights.sum() if ef_weights.sum() > 0 else ef_weights

            ef_weights_dict[layer_idx] = {
                "function_names": layer.elementary_functions,
                "raw_weights": ef_weights,
                "normalized_weights": ef_weights_normalized,
                "in_features": layer.in_features,
                "out_features": layer.out_features
            }

            # 打印权重信息
            tqdm.write(f"  - 初等函数列表: {layer.elementary_functions}")
            for func_name, weight in zip(layer.elementary_functions, ef_weights):
                tqdm.write(f"    {func_name}: {weight:.6f}")

            # 保存CSV
            df = pd.DataFrame({
                "function_name": layer.elementary_functions,
                "raw_weight": ef_weights,
                "normalized_weight": ef_weights_normalized
            })
            csv_path = os.path.join(save_dir, f"ef_weights_layer_{layer_idx}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            tqdm.write(f"  - 权重文件已保存: {csv_path}")

            # 绘制热力图
            if plot_heatmaps:
                top_func_idx = np.argmax(ef_weights)
                top_func_name = layer.elementary_functions[top_func_idx]
                tqdm.write(f"  - 权重最高函数: {top_func_name}")

                # 获取权重矩阵
                top_func_mlp_weights = layer.ef_mlp_linears[top_func_idx].cpu().numpy()

                # 绘制突出重点的热力图（传递是否是第一/最后一幅图的标记）
                plot_stats = plot_ef_weight_heatmap(
                    weight_matrix=top_func_mlp_weights,
                    layer_idx=layer_idx,
                    func_name=top_func_name,
                    save_dir=save_dir,
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    highlight_percentile=95,  # 高亮前5%的权重
                    mask_small_values=True,  # 屏蔽小值
                    small_value_threshold=0.01,  # 小值阈值
                    is_first_plot=is_first,
                    is_last_plot=is_last
                )

                # 将统计信息写入日志
                tqdm.write(
                    f"  - 95%权重分布范围: [{plot_stats['weight_range_95'][0]:.6f}, {plot_stats['weight_range_95'][1]:.6f}]")
                if is_first:
                    tqdm.write(f"  - 第一幅图中高权重最多的5列: {plot_stats['top_5_cols']}")
                if is_last:
                    tqdm.write(f"  - 最后一幅图高权重最多的5行: {plot_stats['top_5_rows']}")

            layer_elapsed = time.time() - layer_start_time
            tqdm.write(f"[层 {current_layer_step}/{total_steps}] 处理完成 (耗时: {layer_elapsed:.2f}秒)")

    # 步骤4: 保存汇总文件
    current_step += 1
    print(f"\n[步骤 {current_step + 1}/{total_steps + 2}] 生成汇总文件...")
    start_time = time.time()

    summary_data = []
    for layer_idx, layer_data in ef_weights_dict.items():
        for func_name, raw_w, norm_w in zip(layer_data["function_names"],
                                            layer_data["raw_weights"],
                                            layer_data["normalized_weights"]):
            summary_data.append({
                "layer_index": layer_idx,
                "function_name": func_name,
                "raw_weight": raw_w,
                "normalized_weight": norm_w
            })

    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(save_dir, "ef_weights_all_layers_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')

    # 保存统计信息到CSV
    if plot_heatmaps and all_plot_stats:
        stats_data = []
        for stats in all_plot_stats:
            stats_data.append({
                "layer_index": stats['layer_idx'],
                "function_name": stats['func_name'],
                "weight_95_range_lower": stats['weight_range_95'][0],
                "weight_95_range_upper": stats['weight_range_95'][1],
                "highlight_threshold": stats['highlight_threshold'],
                "top_5_columns": ','.join(map(str, stats['top_5_cols'])),
                "top_5_columns_counts": ','.join(map(str, stats['top_5_cols_counts'])),
                "top_5_rows": ','.join(map(str, stats['top_5_rows'])),
                "top_5_rows_counts": ','.join(map(str, stats['top_5_rows_counts']))
            })

        stats_df = pd.DataFrame(stats_data)
        stats_csv_path = os.path.join(save_dir, "weight_distribution_stats.csv")
        stats_df.to_csv(stats_csv_path, index=False, encoding='utf-8')
        print(f"[保存] 权重分布统计文件已保存至: {stats_csv_path}")

    elapsed_time = time.time() - start_time
    print(f"[完成] 汇总文件生成完成 (耗时: {elapsed_time:.2f}秒)")
    print(f"[保存] 所有层权重汇总已保存至: {summary_csv_path}")

    # ========== 打印最终统计汇总 ==========
    print("\n" + "=" * 80)
    print("📊 权重分布统计汇总")
    print("=" * 80)
    if plot_heatmaps:
        # 第一幅图统计
        if first_plot_stats:
            print(f"\n🔹 第一幅图 (Layer {first_plot_stats['layer_idx']} - {first_plot_stats['func_name']}):")
            print(
                f"   - 95%权重分布横坐标范围: [{first_plot_stats['weight_range_95'][0]:.6f}, {first_plot_stats['weight_range_95'][1]:.6f}]")
            print(f"   - 中高权重分布最多的5列维度序号: {first_plot_stats['top_5_cols']}")
            print(f"   - 对应列的高权重数量: {first_plot_stats['top_5_cols_counts']}")

        # 最后一幅图统计
        if last_plot_stats:
            print(f"\n🔹 最后一幅图 (Layer {last_plot_stats['layer_idx']} - {last_plot_stats['func_name']}):")
            print(
                f"   - 95%权重分布横坐标范围: [{last_plot_stats['weight_range_95'][0]:.6f}, {last_plot_stats['weight_range_95'][1]:.6f}]")
            print(f"   - 高权重分布最多的5行维度序号: {last_plot_stats['top_5_rows']}")
            print(f"   - 对应行的高权重数量: {last_plot_stats['top_5_rows_counts']}")

        # 所有图的95%分布范围
        print(f"\n🔹 所有图的95%权重分布范围:")
        for stats in all_plot_stats:
            print(f"   Layer {stats['layer_idx']} ({stats['func_name']}): "
                  f"[{stats['weight_range_95'][0]:.6f}, {stats['weight_range_95'][1]:.6f}]")

    # 最终完成提示
    print("\n" + "=" * 60)
    print("权重提取流程执行完毕！")
    print("=" * 60)
    print(f"[统计信息]")
    print(f"  - 处理层数: {total_steps} 层")
    print(f"  - 生成文件数: {total_steps} 个单层权重文件 + 1 个汇总文件")
    if plot_heatmaps:
        print(f"  - 生成热力图数: {total_steps} 张")
        print(f"  - 生成统计文件数: 1 个权重分布统计文件")
    print(f"  - 所有文件保存至: {save_dir}")
    print("=" * 60)

    return ef_weights_dict


# ========== 运行示例 ==========
if __name__ == "__main__":
    # 记录总开始时间
    total_start_time = time.time()

    MODEL_PATH = "/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/Last_SymbolicKAN_Para_1_fold_4L_-524-524-524_15_2.pt"
    LAYERS_HIDDEN = [33, 524, 524, 524, 70]
    ELEMENTARY_FUNCTIONS = ['silu', 'relu', 'tanh', 'sigmoid', 'abs', 'identity']
    SAVE_DIR = "/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_SymbolicParam"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 执行主函数
    ef_weights = extract_ef_weights_per_layer(
        model_path=MODEL_PATH,
        layers_hidden=LAYERS_HIDDEN,
        elementary_functions=ELEMENTARY_FUNCTIONS,
        save_dir=SAVE_DIR,
        device=DEVICE,
        plot_heatmaps=True
    )

    # 总耗时统计
    total_elapsed_time = time.time() - total_start_time
    hours = int(total_elapsed_time // 3600)
    minutes = int((total_elapsed_time % 3600) // 60)
    seconds = total_elapsed_time % 60
    print(f"\n总执行时间: {hours}小时 {minutes}分钟 {seconds:.2f}秒")
