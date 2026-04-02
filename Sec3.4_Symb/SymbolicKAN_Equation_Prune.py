import torch
import torch.nn as nn
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from SymbolicKAN_Finetune import SymbolicKAN, DEFAULT_ELEMENTARY_FUNCTIONS, SUPPORTED_ELEMENTARY_FUNCTIONS

# ==================== 数据集加载（你项目中已有的 MyKANnetLoader） ====================
import MyKANnetLoader


def prune_symbolic_kan_layer(layer, topk_base=10, topk_ef=10):
    """
    对单层SymbolicKANLinear进行结构化剪枝（基础分支 & 初等函数分支 独立剪枝，自动安全限制）
    :param layer: SymbolicKANLinear层实例
    :param topk_base: 基础权重 base_weight 保留的最大权重数量
    :param topk_ef: 所有初等函数线性变换权重保留的最大权重数量
    """
    device = layer.base_weight.device
    in_features = layer.base_weight.shape[1]

    # 安全校验：自动限制不超过输入维度，永不越界
    safe_topk_base = min(topk_base, in_features)
    safe_topk_ef = min(topk_ef, in_features)

    # ==================== 1. 独立剪枝 Base 分支权重 (out_features, in_features) ====================
    base_weight = layer.base_weight.data
    base_abs = torch.abs(base_weight)
    base_topk_val, base_topk_idx = torch.topk(base_abs, k=safe_topk_base, dim=1)

    base_mask = torch.zeros_like(base_weight, device=device)
    base_mask.scatter_(1, base_topk_idx, 1.0)
    layer.base_weight.data = base_weight * base_mask

    # ==================== 2. 独立剪枝 所有初等函数的线性变换权重 ====================
    for func_idx in range(layer.num_ef):
        ef_linear = layer.ef_mlp_linears[func_idx].data
        ef_abs = torch.abs(ef_linear)
        ef_topk_val, ef_topk_idx = torch.topk(ef_abs, k=safe_topk_ef, dim=1)

        ef_mask = torch.zeros_like(ef_linear, device=device)
        ef_mask.scatter_(1, ef_topk_idx, 1.0)
        layer.ef_mlp_linears[func_idx].data = ef_linear * ef_mask

    print(f"✅ 单层剪枝完成：Base分支安全Top-{safe_topk_base} | EF分支安全Top-{safe_topk_ef} (输入维度={in_features})")


def prune_symbolic_kan_model(model, topk_base=10, topk_ef=10):
    """
    对整个SymbolicKAN模型进行逐层剪枝（基础分支 & 初等函数分支 独立控制）
    """
    print(f"\n===== 开始对SymbolicKAN进行结构化剪枝 =====")
    print(f"📌 基础分支保留 Top-{topk_base}")
    print(f"📌 初等函数分支保留 Top-{topk_ef}")
    for idx, layer in enumerate(model.layers):
        print(f"\n正在剪枝第 {idx + 1}/{len(model.layers)} 层...")
        prune_symbolic_kan_layer(layer, topk_base=topk_base, topk_ef=topk_ef)

    total_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum((p == 0).sum().item() for p in model.parameters())
    print(f"\n===== 剪枝完成 =====")
    print(f"模型总参数: {total_params:,}")
    print(f"被置0的参数: {pruned_params:,}")
    print(f"剩余有效参数: {total_params - pruned_params:,}")
    return model


# ==================== 计算每个输出维度的L1误差 ====================
def calculate_per_dimension_l1_loss(model, test_loader, device, output_dim=70):
    model.eval()
    per_dim_sum_loss = torch.zeros(output_dim, device=device)
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            outputs = outputs.squeeze(1)
            targets = targets.squeeze(1)

            abs_error = torch.abs(outputs - targets)
            per_dim_sum_loss += torch.sum(abs_error, dim=0)
            total_samples += inputs.size(0)

    per_dim_avg_loss = (per_dim_sum_loss / total_samples).cpu().numpy()
    return per_dim_avg_loss


# ==================== 测试工具函数 ====================
def calculate_test_l1_loss(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
            total_loss += batch_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    average_loss = total_loss / total_samples
    return average_loss


def count_model_parameters(model, count_non_zero_only=False):
    if hasattr(model, 'module'):
        model = model.module

    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if count_non_zero_only:
            param_non_zero = torch.count_nonzero(param).item()
            total_params += param_non_zero
            if param.requires_grad:
                trainable_params += param_non_zero
        else:
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
    return total_params, trainable_params


def test_inference_and_visualize(test_loader, model, device, save_fig_path=None):
    model.eval()
    total_infer_time = 0
    num_batches = 0

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            inputs, label = data
            inputs, label = inputs.to(device), label.to(device)

            inputs = inputs.squeeze(1)
            label = label.squeeze(1)

            torch.cuda.synchronize()
            start_time = time.time()
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            torch.cuda.synchronize()
            end_time = time.time()

            infer_time = end_time - start_time
            total_infer_time += infer_time
            num_batches += 1

            if idx == 0:
                outputs1 = outputs[20].cpu().numpy()
                label1 = label[20].cpu().numpy()

                plt.figure(figsize=(10, 5))
                plt.plot(range(outputs1.shape[0]), outputs1, 'b-o', label='Model Output')
                plt.plot(range(label1.shape[0]), label1, 'r-x', label='Ground Truth')
                plt.ylim(-2.5, 2.5)
                plt.title('Pruned Model Prediction vs Ground Truth')
                plt.legend()
                plt.grid(True)

                if save_fig_path:
                    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
                    print(f"可视化图像已保存: {save_fig_path}")
                plt.show()
            break

    avg_infer_time = total_infer_time / num_batches
    print(f"\n平均单批次推理时间: {avg_infer_time:.6f} s")
    return avg_infer_time


def save_prune_test_report(report_data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    print(f"\n剪枝测试报告已保存: {save_path}")


# ==================== 主函数 ====================
def parse_prune_args():
    parser = argparse.ArgumentParser(description='SymbolicKAN 结构化剪枝 + 测试（多卡DDP）| 基础分支/初等函数分支 独立剪枝')
    parser.add_argument('--model_ckpt', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/Last_SymbolicKAN_Para_1_fold_4L_-524-524-524_15_2.pt")
    parser.add_argument('--input_nc', type=int, default=33)
    parser.add_argument('--output_nc', type=int, default=70)

    # ==================== 核心修改：独立剪枝参数 ====================
    parser.add_argument('--topk_base', type=int, default=524, help="基础权重 base_weight 保留的Top-K数量")
    parser.add_argument('--topk_ef', type=int, default=400, help="初等函数线性变换权重保留的Top-K数量")

    parser.add_argument('--save_path', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_Output_Input/Pruned_SymbolicKAN_Para_base524_ef400.pt")
    parser.add_argument('--elementary_functions', type=str, nargs='+', default=DEFAULT_ELEMENTARY_FUNCTIONS)

    parser.add_argument('--data_root', type=str, default="/media/aiofm/F/20250723_k-fold-cross-validation-KAN/1-fold")
    parser.add_argument('--batchSize', type=int, default=1280)
    parser.add_argument('--test_report_path', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_Output_Input/prune_test_report_seperate_base524_ef400.json")
    parser.add_argument('--visualize_save_path', type=str, default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_Output_Input/prune_visualization_seperate_base524_ef400.png")

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    return parser.parse_args()


def main_prune_and_test():
    opt = parse_prune_args()

    if opt.local_rank != -1:
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = (opt.local_rank == -1) or (opt.local_rank == 0)
    criterion = nn.L1Loss()

    layers_hidden = [opt.input_nc, 524, 524, 524, opt.output_nc]
    model = SymbolicKAN(layers_hidden=layers_hidden, elementary_functions=opt.elementary_functions).to(device)

    if is_main_process:
        print(f"加载原始模型: {opt.model_ckpt}")
    checkpoint = torch.load(opt.model_ckpt, map_location=device)
    model.load_state_dict(checkpoint['parameter'])

    if opt.local_rank != -1 and torch.cuda.device_count() > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    model_for_prune = model.module if hasattr(model, 'module') else model

    if is_main_process:
        print("\n===== 原始模型统计 =====")
        ori_total, _ = count_model_parameters(model)
        ori_non_zero, _ = count_model_parameters(model, count_non_zero_only=True)
        print(f"总参数: {ori_total:,} | 非零参数: {ori_non_zero:,}")

    train_iter, test_iter, _ = MyKANnetLoader.load_dataset(opt)
    ori_loss = calculate_test_l1_loss(model, test_iter, criterion, device)

    if is_main_process:
        print(f"原始模型测试L1损失: {ori_loss:.6f}")
        print("\n===== 原始模型 70个输出维度误差分析 =====")
        ori_per_dim_loss = calculate_per_dimension_l1_loss(model, test_iter, device, output_dim=70)
        max_error_idx = np.argmax(ori_per_dim_loss)
        max_error_val = ori_per_dim_loss[max_error_idx]
        # 新增：计算损失最大维度的平均损失（占总平均损失比例）
        max_dim_loss_ratio = max_error_val / ori_loss * 100
        print(f"原始模型 → 误差最大的输出维度：第 {max_error_idx} 个")
        print(f"该维度平均L1误差：{max_error_val:.6f}")
        print(f"该维度损失占模型总平均损失比例：{max_dim_loss_ratio:.2f}%")

    # ==================== 执行独立剪枝（核心调用） ====================
    model_for_prune = prune_symbolic_kan_model(
        model_for_prune,
        topk_base=opt.topk_base,
        topk_ef=opt.topk_ef
    )

    if is_main_process:
        torch.save(model_for_prune.state_dict(), opt.save_path)
        print(f"\n剪枝模型已保存: {opt.save_path}")

        print("\n===== 剪枝后模型统计 =====")
        pruned_total, _ = count_model_parameters(model)
        pruned_non_zero, _ = count_model_parameters(model, count_non_zero_only=True)
        zero_params = pruned_total - pruned_non_zero
        compress_rate = (ori_non_zero - pruned_non_zero) / ori_non_zero

        print(f"总参数: {pruned_total:,}")
        print(f"非零参数: {pruned_non_zero:,}")
        print(f"置零参数: {zero_params:,}")
        print(f"有效参数压缩率: {compress_rate:.2%}")

    pruned_loss = calculate_test_l1_loss(model, test_iter, criterion, device)

    if is_main_process:
        loss_change = (pruned_loss - ori_loss) / ori_loss * 100
        print(f"剪枝后L1损失: {pruned_loss:.6f} | 损失变化: {loss_change:.2f}%")

        print("\n===== 剪枝后模型 70个输出维度误差分析 =====")
        pruned_per_dim_loss = calculate_per_dimension_l1_loss(model, test_iter, device, output_dim=70)
        max_error_idx_pruned = np.argmax(pruned_per_dim_loss)
        max_error_val_pruned = pruned_per_dim_loss[max_error_idx_pruned]
        # 新增：剪枝后损失最大维度统计
        max_dim_loss_ratio_pruned = max_error_val_pruned / pruned_loss * 100
        print(f"剪枝后模型 → 误差最大的输出维度：第 {max_error_idx_pruned} 个")
        print(f"该维度平均L1误差：{max_error_val_pruned:.6f}")
        print(f"该维度损失占模型总平均损失比例：{max_dim_loss_ratio_pruned:.2f}%")

        avg_infer_time = test_inference_and_visualize(test_iter, model, device, save_fig_path=opt.visualize_save_path)

        # 报告增强：记录两个独立剪枝参数 + 最大损失维度详细信息
        report = {
            "prune_config": {
                "topk_base": opt.topk_base,
                "topk_ef": opt.topk_ef
            },
            "original_model": {
                "total_params": ori_total,
                "non_zero": ori_non_zero,
                "loss": round(ori_loss, 6),
                "max_error_dim": int(max_error_idx),
                "max_error_value": round(float(max_error_val), 6),
                "max_dim_loss_ratio%": round(float(max_dim_loss_ratio), 2)
            },
            "pruned_model": {
                "non_zero": pruned_non_zero,
                "compress_rate": round(compress_rate, 4),
                "loss": round(pruned_loss, 6),
                "loss_change%": round(loss_change, 2),
                "max_error_dim": int(max_error_idx_pruned),
                "max_error_value": round(float(max_error_val_pruned), 6),
                "max_dim_loss_ratio%": round(float(max_dim_loss_ratio_pruned), 2)
            },
            "inference": {"avg_batch_time": round(avg_infer_time, 6)}
        }
        save_prune_test_report(report, opt.test_report_path)

        print("\n" + "=" * 60)
        print("✅ 多GPU独立剪枝+测试完成！")
        print(f"Base保留Top-{opt.topk_base} | EF保留Top-{opt.topk_ef} | 压缩率: {compress_rate:.2%} | 损失变化: {loss_change:.2f}%")
        print("=" * 60)


if __name__ == '__main__':
    main_prune_and_test()
