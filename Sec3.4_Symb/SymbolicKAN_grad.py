import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from torch.distributed import get_rank, is_initialized
import pandas as pd

# ========== 1. 导入原有SymbolicKAN定义（保持和训练代码一致） ==========
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

    def forward(self, x: torch.Tensor):
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


# ========== 2. 工具函数 ==========
def is_main_process():
    return get_rank() == 0 if is_initialized() else True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_ckpt', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/Last_SymbolicKAN_Para_1_fold_4L_-524-524-524_15_2.pt",
                        help='训练好的SymbolicKAN模型权重路径')
    parser.add_argument('--elementary_functions', type=str, nargs='+',
                        default=DEFAULT_ELEMENTARY_FUNCTIONS,
                        help=f'训练时使用的初等函数列表')
    parser.add_argument('--data_root', type=str,
                        default="/media/aiofm/F/20250723_k-fold-cross-validation-KAN/1-fold",
                        help='HDF5数据集根目录')
    parser.add_argument('--input_nc', type=int, default=33, help='输入特征维度')
    parser.add_argument('--output_nc', type=int, default=70, help='输出特征维度')
    parser.add_argument('--batchSize', type=int, default=10240, help='计算梯度时的批次大小')
    parser.add_argument('--save_path', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_SymbolicParam",
                        help='梯度幅值结果保存目录')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    # 新增：热力图相关参数
    parser.add_argument('--heatmap_figsize', type=int, nargs=2, default=[20, 15],
                        help='热力图尺寸 (宽度, 高度)')
    parser.add_argument('--heatmap_dpi', type=int, default=300, help='热力图分辨率')
    parser.add_argument('--heatmap_tick_interval', type=int, default=1,
                        help='热力图刻度显示间隔（1表示显示所有特征）')
    return parser.parse_args()


# ========== 3. 核心梯度计算函数 ==========
def calculate_average_gradient_magnitude(model, test_loader, device):
    """
    计算每个输出相对于每个输入的平均梯度幅值
    返回:
        gradient_magnitude: shape [output_dim, input_dim]，每个输出对每个输入的平均梯度幅值
        sample_count: 参与计算的样本总数
    """
    # 兼容DDP和非DDP模式获取原始模型
    raw_model = model.module if is_initialized() else model
    # 保持train模式但冻结参数梯度
    raw_model.train()
    for param in raw_model.parameters():
        param.requires_grad = False

    # 初始化梯度累加器和样本计数器
    input_dim = raw_model.layers[0].in_features
    output_dim = raw_model.layers[-1].out_features
    total_gradient_magnitude = torch.zeros((output_dim, input_dim), device=device)
    sample_count = 0

    for batch_idx, (X, _) in enumerate(test_loader):
        X = X.to(device, non_blocking=True)
        X.requires_grad = True  # 启用输入的梯度计算

        # 前向传播
        outputs = model(X)
        batch_size = X.shape[0]

        # 对每个输出维度计算梯度
        for output_idx in range(output_dim):
            # 清空之前的梯度
            if X.grad is not None:
                X.grad.zero_()

            # 计算当前输出维度对输入的梯度
            # 修复：适配可能的3维输出（batch, 1, output_dim）
            if len(outputs.shape) == 3:
                output_sum = outputs[:, 0, output_idx].sum()
            else:
                output_sum = outputs[:, output_idx].sum()
            output_sum.backward(retain_graph=True, create_graph=False)

            # 获取梯度并计算幅值（绝对值）
            grad = X.grad.detach()  # shape [batch_size, input_dim] or [batch_size, 1, input_dim]

            # 关键修复：确保梯度维度正确，压缩多余的维度
            # 将梯度展平为 [batch_size, input_dim]
            grad = grad.view(batch_size, -1)
            grad_magnitude = torch.abs(grad)  # 梯度幅值

            # 计算当前批次的梯度和，并确保维度为 [input_dim]
            grad_sum = grad_magnitude.sum(dim=0).squeeze()  # 移除多余的维度

            # 确保维度匹配后再累加
            if grad_sum.dim() == 0:
                # 处理极端情况：input_dim=1时
                grad_sum = grad_sum.unsqueeze(0)

            # 累加当前批次的梯度幅值
            total_gradient_magnitude[output_idx] += grad_sum

        # 更新样本计数
        sample_count += batch_size

        # 打印进度
        if is_main_process() and (batch_idx + 1) % 10 == 0:
            print(f"Processed batch {batch_idx + 1}/{len(test_loader)}, samples processed: {sample_count}")

        # 释放内存
        del X.grad
        torch.cuda.empty_cache()

    # 计算平均梯度幅值
    if sample_count > 0:
        average_gradient_magnitude = total_gradient_magnitude / sample_count
    else:
        raise ValueError("No samples found in test loader!")

    return average_gradient_magnitude.cpu().numpy(), sample_count


# ========== 4. 结果保存函数（优化热力图） ==========
def save_gradient_results(gradient_magnitude, sample_count, save_dir, input_dim=33, output_dim=70,
                          heatmap_figsize=(20, 15), heatmap_dpi=300, heatmap_tick_interval=1):
    """
    保存梯度幅值结果：
    1. 保存为numpy数组（便于后续加载使用）
    2. 保存为CSV文件（便于查看和分析）
    3. 保存为可视化热力图（蓝到红渐变）
    4. 保存每个输出对输入梯度的排序结果（从大到小）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存原始numpy数组
    np.save(os.path.join(save_dir, "average_gradient_magnitude.npy"), gradient_magnitude)

    # 2. 保存原始CSV文件
    # 创建行列标签
    input_labels = [f"input_{i + 1}" for i in range(input_dim)]
    output_labels = [f"output_{i + 1}" for i in range(output_dim)]
    df = pd.DataFrame(gradient_magnitude, index=output_labels, columns=input_labels)
    df.to_csv(os.path.join(save_dir, "average_gradient_magnitude.csv"), float_format='%.6f')

    # 3. 保存统计信息
    stats = {
        "sample_count": sample_count,
        "mean_gradient_magnitude": float(np.mean(gradient_magnitude)),
        "max_gradient_magnitude": float(np.max(gradient_magnitude)),
        "min_gradient_magnitude": float(np.min(gradient_magnitude)),
        "std_gradient_magnitude": float(np.std(gradient_magnitude))
    }
    with open(os.path.join(save_dir, "gradient_stats.txt"), "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    # 4. 对每个输出维度，按输入梯度幅值从大到小排序
    sorted_results = []
    for output_idx in range(output_dim):
        # 获取当前输出对所有输入的梯度值
        grad_values = gradient_magnitude[output_idx]

        # 获取排序后的索引（从大到小）
        sorted_indices = np.argsort(grad_values)[::-1]

        # 构建当前输出的排序结果
        output_result = {
            "output_label": f"output_{output_idx + 1}",
            "sorted_inputs": [],
            "sorted_values": []
        }

        for idx in sorted_indices:
            output_result["sorted_inputs"].append(f"input_{idx + 1}")
            output_result["sorted_values"].append(grad_values[idx])

        sorted_results.append(output_result)

    # 5. 保存排序后的详细结果（CSV格式）
    sorted_data = []
    for res in sorted_results:
        for i, (inp, val) in enumerate(zip(res["sorted_inputs"], res["sorted_values"])):
            sorted_data.append({
                "output": res["output_label"],
                "rank": i + 1,  # 排名（1为最大）
                "input": inp,
                "gradient_magnitude": val
            })
    sorted_df = pd.DataFrame(sorted_data)
    sorted_df.to_csv(os.path.join(save_dir, "sorted_gradient_magnitude.csv"), index=False, float_format='%.6f')

    # 6. 保存每个输出的Top-N汇总（便于快速查看）
    top_n = 10  # 可根据需要调整显示前N个
    top_summary = []
    for res in sorted_results:
        top_summary.append({
            "output": res["output_label"],
            f"top_{top_n}_inputs": ", ".join(res["sorted_inputs"][:top_n]),
            f"top_{top_n}_values": ", ".join([f"{v:.6f}" for v in res["sorted_values"][:top_n]]),
            "max_gradient_input": res["sorted_inputs"][0],
            "max_gradient_value": res["sorted_values"][0],
            "min_gradient_input": res["sorted_inputs"][-1],
            "min_gradient_value": res["sorted_values"][-1]
        })
    top_summary_df = pd.DataFrame(top_summary)
    top_summary_df.to_csv(os.path.join(save_dir, f"gradient_top{top_n}_summary.csv"), index=False)

    # 7. 优化版热力图（蓝到红渐变）
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        # 设置中文字体（可选，根据需要调整）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建画布
        fig, ax = plt.subplots(figsize=heatmap_figsize, dpi=heatmap_dpi)

        # 定义颜色映射：从蓝色到红色（低->高）
        # 使用coolwarm的反向，或者自定义蓝红渐变
        cmap = plt.cm.RdBu_r  # RdBu_r: 红色(高) -> 蓝色(低)，正好符合需求
        # 也可以使用自定义渐变：
        # cmap = mpl.colors.LinearSegmentedColormap.from_list('blue_to_red',
        #                                                   ['#0000FF', '#FFFFFF', '#FF0000'],
        #                                                   N=256)

        # 绘制热力图
        im = ax.imshow(gradient_magnitude, cmap=cmap, aspect='auto')

        # 设置标题
        ax.set_title(
            f'Average Gradient Magnitude (Output vs Input)\nTotal Samples: {sample_count:,}',
            fontsize=16, fontweight='bold', pad=20
        )

        # 设置轴标签
        ax.set_xlabel('Input Features', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Output Features', fontsize=14, fontweight='bold', labelpad=10)

        # 设置刻度
        # X轴（输入特征）
        x_ticks = range(0, input_dim, heatmap_tick_interval)
        x_tick_labels = [f"Input {i + 1}" for i in x_ticks]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=8)

        # Y轴（输出特征）
        y_ticks = range(0, output_dim, heatmap_tick_interval)
        y_tick_labels = [f"Output {i + 1}" for i in y_ticks]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels, fontsize=8)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Average Gradient Magnitude', fontsize=12, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(labelsize=10)

        # 添加网格线（可选）
        ax.grid(False)  # 关闭网格线，使热力图更清晰

        # 调整布局
        plt.tight_layout()

        # 保存热力图
        heatmap_path = os.path.join(save_dir, "gradient_magnitude_heatmap_blue_to_red.png")
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=heatmap_dpi)
        plt.close()

        print(f"Optimized heatmap saved to: {heatmap_path}")

    except ImportError:
        print("Matplotlib not found, skipping heatmap generation")
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")

    # 打印保存信息
    print(f"\nResults saved to: {save_dir}")
    print(f"- 原始梯度矩阵(npy): average_gradient_magnitude.npy")
    print(f"- 原始梯度矩阵(csv): average_gradient_magnitude.csv")
    print(f"- 统计信息: gradient_stats.txt")
    print(f"- 排序后详细结果: sorted_gradient_magnitude.csv")
    print(f"- Top-{top_n}汇总结果: gradient_top{top_n}_summary.csv")
    print(f"- 优化版热力图: gradient_magnitude_heatmap_blue_to_red.png (if matplotlib is available)")


# ========== 5. 主函数 ==========
def main():
    opt = parse_args()

    # 初始化设备
    if opt.local_rank != -1:
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print(f"Using device: {device}")
        print(f"Distributed training: {'Enabled' if is_initialized() else 'Disabled'}")

    # 1. 加载模型
    layers_hidden = [opt.input_nc, 524, 524, 524, opt.output_nc]
    model = SymbolicKAN(
        layers_hidden=layers_hidden,
        elementary_functions=opt.elementary_functions
    ).to(device)

    # 加载训练好的权重
    checkpoint = torch.load(opt.model_ckpt, map_location=device)
    if is_initialized():
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=True
        )
        model.module.load_state_dict(checkpoint['parameter'])
    else:
        model.load_state_dict(checkpoint['parameter'])

    if is_main_process():
        print(f"Model loaded from: {opt.model_ckpt}")
        print(f"Model structure: input_dim={opt.input_nc}, hidden_layers=[524,524,524], output_dim={opt.output_nc}")

    # 2. 加载测试数据集
    import MyKANnetLoader  # 你的数据集加载器
    _, test_iter, _ = MyKANnetLoader.load_dataset(opt)

    # 3. 计算平均梯度幅值
    if is_main_process():
        print("\n=== Starting gradient magnitude calculation ===")
    gradient_magnitude, sample_count = calculate_average_gradient_magnitude(model, test_iter, device)

    # 4. 保存结果（仅主进程执行）
    if is_main_process():
        print(f"\n=== Calculation completed ===")
        print(f"Total samples processed: {sample_count}")
        print(f"Gradient magnitude matrix shape: {gradient_magnitude.shape}")
        print(f"Mean gradient magnitude: {np.mean(gradient_magnitude):.6f}")
        print(f"Max gradient magnitude: {np.max(gradient_magnitude):.6f}")
        print(f"Min gradient magnitude: {np.min(gradient_magnitude):.6f}")

        save_gradient_results(
            gradient_magnitude=gradient_magnitude,
            sample_count=sample_count,
            save_dir=opt.save_path,
            input_dim=opt.input_nc,
            output_dim=opt.output_nc,
            heatmap_figsize=opt.heatmap_figsize,
            heatmap_dpi=opt.heatmap_dpi,
            heatmap_tick_interval=opt.heatmap_tick_interval
        )


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
