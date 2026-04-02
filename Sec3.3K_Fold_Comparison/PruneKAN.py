import efficient_kan.kan as E_kan

import torch
import torch.nn as nn
import MyKANnetLoader
import argparse
import functools
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import joblib
from sklearn.decomposition import PCA
from scipy.signal import correlate
import time
from typing import Dict, List, Tuple


def zernikeGen(N, coeff, ZernPoly36, **kwargs):
    # Generating the Zernike Phase representation.
    num_coeff = coeff.shape[1]

    # Setting up 2D grid
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), np.linspace(-1, 1, N, endpoint=True))

    zern_out = np.zeros((N, N, coeff.shape[1]))

    # 获取当前工作目录
    for i in range(0, num_coeff):
        zern_out[:, :, i] = coeff[0, i] * ZernPoly36[:, :, i + 3]

    return zern_out


def ComputeOTF(a):
    ZernPoly36 = np.load('/home/aiofm/PycharmProjects/MyKANNet/36—128ZernPoly.npy')

    # 采样网格数
    N = 128

    zernike_stack = zernikeGen(N, a, ZernPoly36)
    # 计算相位
    Fai = np.sum(zernike_stack, axis=2)

    # 孔径函数
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), np.linspace(-1, 1, N, endpoint=True))
    mask = np.sqrt(x_grid ** 2 + y_grid ** 2) <= 1

    # 根据傅里叶变换得到对偶定理，F(F(w()))计算OTF
    wave = mask * np.exp(1j * 2 * np.pi * Fai)
    # wave进行归一化,离散Parseval定理np.sum(PSF)/(128**2)=np.sum(np.abs(wave)**2)
    p = np.sum(np.abs(wave) ** 2)
    wave = wave * (((1 / 128 ** 2) / p) ** 0.5)

    cor = correlate(wave, wave, mode='same') * N ** 2
    cor = cor[::-1, ::-1]

    return cor


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str,
                    default="/media/aiofm/F/20250723_k-fold-cross-validation-KAN/4-fold", help='数据集的根目录')
parser.add_argument('--input_nc', type=int, default=33, help='输入维度的通道数量')
parser.add_argument('--output_nc', type=int, default=70, help='输出维度的通道数量')
parser.add_argument('--batchSize', type=int, default=64, help='一次训练载入的数据量')
parser.add_argument('--learn_rate', type=float, default=0.00005, help='初始学习率')
parser.add_argument('--num_epochs', type=int, default=300, help='训练的轮数')

# 保存路径（确保主进程有写入权限）
parser.add_argument('--SaveTrainLossPath', type=str,
                    default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_Kan4LayerParam_L1Regu/record_train_loss_7_fold-524-524-524_15_2.txt")
parser.add_argument('--SaveValLossPath', type=str,
                    default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_Kan4LayerParam_L1Regu/record_val_loss_7_fold-524-524-524_15_2.txt")
parser.add_argument('--SavePara', type=str,
                    default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_Kan4LayerParam_L1Regu/Kan_Para_7_fold-524-524-524_15_2.pt")
parser.add_argument('--SaveLastPara', type=str,
                    default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_Kan4LayerParam_L1Regu/Last_Kan_Para_7_fold-524-524-524_15_2.pt")

parser.add_argument('--num_print', type=int, default=10, help='每过num_print轮训练打印一次损失')
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

# 剪枝相关参数（仅保留阈值剪枝，移除比例剪枝）
parser.add_argument('--prune_threshold', type=float, default=0.005,
                    help='剪枝阈值（绝对值小于该值的参数将被置零）')
parser.add_argument('--prune_target', type=str, default='all',
                    choices=['splines', 'base_weight', 'spline_scaler', 'all', 'linear'],
                    help='剪枝目标：splines(样条权重), base_weight(基权重), spline_scaler(缩放因子), all(全部KAN参数), linear(普通线性层)')
parser.add_argument('--layer_thresholds', type=str, default=None,
                    help='分层阈值配置（JSON字符串），例如：{"layers.0.spline_weight":0.005, "layers.1.base_weight":0.02}')
parser.add_argument('--save_pruned_model', type=str,
                    default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_PruneParam/Pruned_Kan_Para_4_fold_4L_-524-524-524_15_2.pt",
                    help='保存剪枝后模型的路径')

opt = parser.parse_args()

# 设备配置
if opt.local_rank != -1:
    torch.cuda.set_device(opt.local_rank)
    device = torch.device("cuda", opt.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 解析分层阈值（可选）
threshold_dict = {}
if opt.layer_thresholds:
    import json

    try:
        threshold_dict = json.loads(opt.layer_thresholds)
        if opt.local_rank in [-1, 0]:  # 仅主进程打印
            print(f"Loaded layer-specific thresholds: {threshold_dict}")
    except json.JSONDecodeError:
        if opt.local_rank in [-1, 0]:
            print("Warning: Invalid JSON format for layer_thresholds, using global threshold instead")

# 判断是否为主进程（仅主进程执行打印/保存操作）
is_main_process = (opt.local_rank == -1) or (opt.local_rank == 0)


# ===== 工具函数 =====
def test(test_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            inputs, label = data
            inputs, label = inputs.to(device), label.to(device)
            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            if is_main_process:
                print(f"Inference time (batch {idx}): {end_time - start_time:.6f} seconds")

            # 可视化第20个样本
            if idx == 0 and is_main_process:
                outputs1 = outputs[20].cpu().numpy()
                label1 = label[20].cpu().numpy()
                plt.plot(range(outputs1.shape[1]), outputs1[0, :], color='blue', marker='o', linestyle='-',
                         label='output')
                plt.plot(range(label1.shape[1]), label1[0, :], color='red', marker='x', linestyle='-', label='label')
                plt.ylim(-2.5, 2.5)
                plt.legend()
                plt.show()
            break  # 只测试第一个batch
        if is_main_process:
            print("Test finished")


def calculate_test_l1_loss(model, test_loader, criterion):
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
    """
    统计模型参数
    :param model: 模型
    :param count_non_zero_only: 是否仅统计非零参数（用于剪枝后有效参数统计）
    :return: total_params, trainable_params
    """
    if hasattr(model, 'module'):
        model = model.module

    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        if count_non_zero_only:
            # 仅统计非零参数
            param_non_zero = torch.count_nonzero(param).item()
            total_params += param_non_zero
            if param.requires_grad:
                trainable_params += param_non_zero
        else:
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
    return total_params, trainable_params


# ===== 核心剪枝器（纯阈值剪枝）=====
class KANPruner:
    """基于固定阈值的KAN剪枝器"""

    def __init__(self, model, prune_threshold=0.01, prune_target='splines', layer_thresholds=None):
        self.model = model  # 原始模型（非DDP包装）
        self.global_threshold = prune_threshold
        self.prune_target = prune_target
        self.layer_thresholds = layer_thresholds or {}  # 分层阈值
        self.pruned_info = []
        self.masks = {}
        self.total_pruned_params = 0  # 剪枝的参数总数（置零的参数数）
        self.total_target_params = 0  # 剪枝目标的总参数数
        # 新增：存储损失信息
        self.original_loss = None
        self.pruned_loss = None

    def set_loss_info(self, original_loss, pruned_loss):
        """设置剪枝前后的损失信息"""
        self.original_loss = original_loss
        self.pruned_loss = pruned_loss

    def _get_param_threshold(self, layer_name, param_name):
        """获取参数的剪枝阈值（优先分层阈值，其次全局阈值）"""
        full_param_name = f"{layer_name}.{param_name}"
        return self.layer_thresholds.get(full_param_name, self.global_threshold)

    def _collect_kan_layers(self):
        """收集模型中所有KANLinear层（支持所有变体）"""
        kan_layers = []

        # 递归查找所有KANLinear层
        def recursive_find_kan(module, parent_name=""):
            for name, child in module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                if isinstance(child, E_kan.KANLinear):
                    kan_layers.append((full_name, child))
                elif isinstance(child, (nn.ModuleList, nn.Sequential)):
                    recursive_find_kan(child, full_name)
                else:
                    # 递归查找子模块
                    recursive_find_kan(child, full_name)

        recursive_find_kan(self.model)
        return kan_layers

    def _collect_linear_layers(self):
        """收集模型中所有普通Linear层（如Attention、FiLM中的线性层）"""
        linear_layers = []

        def recursive_find_linear(module, parent_name=""):
            for name, child in module.named_children():
                full_name = f"{parent_name}.{name}" if parent_name else name
                if isinstance(child, nn.Linear) and not isinstance(child, E_kan.KANLinear):
                    linear_layers.append((full_name, child))
                else:
                    recursive_find_linear(child, full_name)

        recursive_find_linear(self.model)
        return linear_layers

    def _create_threshold_mask(self, param_tensor, threshold, param_type):
        """根据阈值创建剪枝掩码"""
        # 计算参数绝对值
        param_abs = torch.abs(param_tensor)

        # 生成掩码：>=阈值保留（1），<阈值剪枝（0）
        if param_type == 'splines':
            # Splines参数：[out, in, grid+spline] → 按in/out维度计算重要性
            importance = torch.norm(param_tensor, p=1, dim=-1)  # [out, in]
            mask = (importance >= threshold).float().to(device)
            mask = mask.unsqueeze(-1).expand(param_tensor.shape)  # 扩展到三维
        else:
            # 普通参数直接按绝对值判断
            mask = (param_abs >= threshold).float().to(device)

        # 统计剪枝信息
        total_params = param_tensor.numel()
        pruned_params = torch.sum(mask == 0).item()
        prune_ratio = pruned_params / total_params if total_params > 0 else 0.0

        # 累计剪枝统计
        self.total_pruned_params += pruned_params
        self.total_target_params += total_params

        return mask, pruned_params, total_params, prune_ratio

    def prune_kan_layers(self):
        """剪枝KANLinear层（纯阈值控制）"""
        kan_layers = self._collect_kan_layers()

        if is_main_process:
            print(f"\n=== Threshold-based Pruning KANLinear Layers (target: {self.prune_target}) ===")
            print(f"Global threshold: {self.global_threshold:.6f}")

        for layer_name, kan_layer in kan_layers:
            # 剪枝spline_weight
            if self.prune_target in ['splines', 'all'] and hasattr(kan_layer, 'spline_weight'):
                splines = kan_layer.spline_weight.data
                threshold = self._get_param_threshold(layer_name, 'spline_weight')

                # 生成掩码并应用
                mask, pruned, total, ratio = self._create_threshold_mask(
                    splines, threshold, 'splines'
                )
                kan_layer.spline_weight.data = splines * mask
                self.masks[f"{layer_name}.spline_weight"] = mask

                # 记录信息
                self.pruned_info.append({
                    'layer_name': layer_name,
                    'param_name': 'spline_weight',
                    'shape': splines.shape,
                    'threshold': threshold,
                    'pruned_params': pruned,
                    'total_params': total,
                    'prune_ratio': ratio
                })

                if is_main_process:
                    print(f"\n{layer_name} - spline_weight:")
                    print(f"  Shape: {splines.shape}")
                    print(f"  Threshold: {threshold:.6f}")
                    print(f"  Pruned: {pruned}/{total} ({ratio:.2%})")

            # 剪枝base_weight
            if self.prune_target in ['base_weight', 'all'] and hasattr(kan_layer, 'base_weight'):
                base_weight = kan_layer.base_weight.data
                threshold = self._get_param_threshold(layer_name, 'base_weight')

                mask, pruned, total, ratio = self._create_threshold_mask(
                    base_weight, threshold, 'base_weight'
                )
                kan_layer.base_weight.data = base_weight * mask
                self.masks[f"{layer_name}.base_weight"] = mask

                self.pruned_info.append({
                    'layer_name': layer_name,
                    'param_name': 'base_weight',
                    'shape': base_weight.shape,
                    'threshold': threshold,
                    'pruned_params': pruned,
                    'total_params': total,
                    'prune_ratio': ratio
                })

                if is_main_process:
                    print(f"\n{layer_name} - base_weight:")
                    print(f"  Shape: {base_weight.shape}")
                    print(f"  Threshold: {threshold:.6f}")
                    print(f"  Pruned: {pruned}/{total} ({ratio:.2%})")

            # 剪枝spline_scaler
            if self.prune_target in ['spline_scaler', 'all'] and hasattr(kan_layer, 'spline_scaler'):
                scaler = kan_layer.spline_scaler.data if hasattr(kan_layer, 'spline_scaler') else None
                if scaler is not None:
                    threshold = self._get_param_threshold(layer_name, 'spline_scaler')

                    mask, pruned, total, ratio = self._create_threshold_mask(
                        scaler, threshold, 'spline_scaler'
                    )
                    kan_layer.spline_scaler.data = scaler * mask
                    self.masks[f"{layer_name}.spline_scaler"] = mask

                    self.pruned_info.append({
                        'layer_name': layer_name,
                        'param_name': 'spline_scaler',
                        'shape': scaler.shape,
                        'threshold': threshold,
                        'pruned_params': pruned,
                        'total_params': total,
                        'prune_ratio': ratio
                    })

                    if is_main_process:
                        print(f"\n{layer_name} - spline_scaler:")
                        print(f"  Shape: {scaler.shape}")
                        print(f"  Threshold: {threshold:.6f}")
                        print(f"  Pruned: {pruned}/{total} ({ratio:.2%})")

        return self.total_pruned_params, self.total_target_params

    def prune_linear_layers(self):
        """剪枝普通Linear层（纯阈值控制）"""
        linear_layers = self._collect_linear_layers()

        if is_main_process:
            print(f"\n=== Threshold-based Pruning Standard Linear Layers ===")
            print(f"Global threshold: {self.global_threshold:.6f}")

        for layer_name, linear_layer in linear_layers:
            # 剪枝权重
            weight = linear_layer.weight.data
            weight_threshold = self._get_param_threshold(layer_name, 'weight')
            weight_mask, weight_pruned, weight_total, weight_ratio = self._create_threshold_mask(
                weight, weight_threshold, 'linear'
            )
            linear_layer.weight.data = weight * weight_mask
            self.masks[f"{layer_name}.weight"] = weight_mask

            # 剪枝偏置（如果有）
            bias_pruned = 0
            bias_total = 0
            bias_ratio = 0
            if linear_layer.bias is not None:
                bias = linear_layer.bias.data
                bias_threshold = self._get_param_threshold(layer_name, 'bias')
                bias_mask, bias_pruned, bias_total, bias_ratio = self._create_threshold_mask(
                    bias, bias_threshold, 'linear'
                )
                linear_layer.bias.data = bias * bias_mask
                self.masks[f"{layer_name}.bias"] = bias_mask

            # 统计该层总剪枝信息
            layer_pruned = weight_pruned + bias_pruned
            layer_total = weight_total + bias_total
            layer_ratio = layer_pruned / layer_total if layer_total > 0 else 0.0

            self.pruned_info.append({
                'layer_name': layer_name,
                'param_name': 'linear_layer',
                'shape': (weight.shape, bias.shape if linear_layer.bias is not None else None),
                'threshold': (weight_threshold, bias_threshold if linear_layer.bias is not None else None),
                'pruned_params': layer_pruned,
                'total_params': layer_total,
                'prune_ratio': layer_ratio
            })

            if is_main_process:
                print(f"\n{layer_name}:")
                print(
                    f"  Weight shape: {weight.shape} | Threshold: {weight_threshold:.6f} | Pruned: {weight_pruned}/{weight_total} ({weight_ratio:.2%})")
                if linear_layer.bias is not None:
                    print(
                        f"  Bias shape: {bias.shape} | Threshold: {bias_threshold:.6f} | Pruned: {bias_pruned}/{bias_total} ({bias_ratio:.2%})")
                print(f"  Total pruned: {layer_pruned}/{layer_total} ({layer_ratio:.2%})")

        return self.total_pruned_params, self.total_target_params

    def prune(self):
        """执行阈值剪枝"""
        # 剪枝KAN层
        kan_pruned, kan_total = self.prune_kan_layers()

        # 如果目标是linear，剪枝普通线性层
        if self.prune_target == 'linear':
            linear_pruned, linear_total = self.prune_linear_layers()
        else:
            linear_pruned, linear_total = 0, 0

        # 打印整体统计（仅主进程）
        if is_main_process:
            print("\n=== Overall Threshold Pruning Summary ===")
            total_pruned = kan_pruned + linear_pruned
            total_params = kan_total + linear_total
            if total_params > 0:
                overall_ratio = total_pruned / total_params
                print(f"Total pruned parameters (zeroed): {total_pruned:,}/{total_params:,} ({overall_ratio:.2%})")
                print(f"Remaining non-zero parameters: {total_params - total_pruned:,} ({1 - overall_ratio:.2%})")
            else:
                print("No parameters were pruned!")

    def save_pruned_model(self, path):
        """保存剪枝后的模型（仅主进程执行）"""
        if is_main_process:
            torch.save(self.model.state_dict(), path)
            print(f"\nPruned model saved to: {path}")

    def export_pruning_report(self, path=None):
        """导出剪枝报告（可选，仅主进程执行），包含剪枝前后的L1损失"""
        if is_main_process and path:
            import json
            total_pruned = sum(item['pruned_params'] for item in self.pruned_info)
            total_params = sum(item['total_params'] for item in self.pruned_info)

            # 计算损失变化率
            loss_change_ratio = 0.0
            loss_change_abs = 0.0
            if self.original_loss is not None and self.pruned_loss is not None:
                loss_change_abs = self.pruned_loss - self.original_loss
                if self.original_loss != 0:
                    loss_change_ratio = (loss_change_abs / self.original_loss) * 100

            report = {
                'global_threshold': self.global_threshold,
                'prune_target': self.prune_target,
                'layer_thresholds': self.layer_thresholds,
                'pruned_info': self.pruned_info,
                'total_pruned_params': total_pruned,
                'total_target_params': total_params,
                'overall_prune_ratio': total_pruned / total_params if total_params > 0 else 0.0,
                # 新增：损失相关信息
                'loss_metrics': {
                    'original_l1_loss': self.original_loss,
                    'pruned_l1_loss': self.pruned_loss,
                    'loss_change_absolute': loss_change_abs,
                    'loss_change_percentage': loss_change_ratio
                }
            }
            with open(path, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"Pruning report saved to: {path}")


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # 仅主进程打印GPU信息
    if is_main_process:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and opt.local_rank != -1:
            print(f'Using {num_gpus} GPUs!')

    # 创建KAN模型
    net = E_kan.KAN([opt.input_nc, 524, 524, 524, opt.output_nc], grid_size=15, spline_order=2)

    # 将模型放入GPU中
    net = net.to(device)

    # DDP包装（仅多GPU时）
    if opt.local_rank != -1 and torch.cuda.device_count() > 1:
        net = nn.parallel.DistributedDataParallel(net, device_ids=[opt.local_rank],
                                                  output_device=opt.local_rank)

    # 加载预训练参数（所有进程都要加载）
    checkpoint = torch.load(opt.SaveLastPara, map_location=device)  # 加载断点
    net.module.load_state_dict(checkpoint['parameter'])  # 加载模型可学习参数

    # 加载数据集
    train_iter, test_iter, val_test = MyKANnetLoader.load_dataset(opt)

    # 定义损失函数
    criterion = torch.nn.L1Loss()

    # 1. 统计原始模型信息（仅主进程）
    if is_main_process:
        print("\n=== Original Model Statistics ===")
        original_total, original_trainable = count_model_parameters(net)
        original_non_zero, _ = count_model_parameters(net, count_non_zero_only=True)
        print(f"Total parameters: {original_total:,}")
        print(f"Trainable parameters: {original_trainable:,}")
        print(f"Non-zero parameters (original): {original_non_zero:,}")

    # 2. 计算剪枝前性能（所有进程都计算，但仅主进程打印）
    if is_main_process:
        print("\nCalculating original model performance...")
    original_loss = calculate_test_l1_loss(net, test_iter, criterion)
    if is_main_process:
        print(f"Original model L1 loss: {original_loss:.6f}")

    # 3. 执行阈值剪枝（所有进程都执行剪枝，但打印仅主进程）
    pruner = KANPruner(
        model=net.module if hasattr(net, 'module') else net,
        prune_threshold=opt.prune_threshold,
        prune_target=opt.prune_target,
        layer_thresholds=threshold_dict
    )
    pruner.prune()

    # 4. 统计剪枝后信息（仅主进程）
    if is_main_process:
        print("\n=== Pruned Model Statistics ===")
        pruned_total, pruned_trainable = count_model_parameters(net)
        pruned_non_zero, _ = count_model_parameters(net, count_non_zero_only=True)

        # 计算有效参数缩减率（非零参数的减少比例）
        param_reduction_ratio = (
                                        original_non_zero - pruned_non_zero) / original_non_zero if original_non_zero > 0 else 0.0

        print(f"Total parameters (unchanged): {pruned_total:,} (100.00% of original)")
        print(f"Trainable parameters (unchanged): {pruned_trainable:,} (100.00% of original)")
        print(
            f"Non-zero parameters (pruned): {pruned_non_zero:,} ({pruned_non_zero / original_non_zero:.2%} of original)")
        print(f"Effective parameter reduction: {param_reduction_ratio:.2%}")

    # 5. 计算剪枝后性能（所有进程都计算，仅主进程打印）
    if is_main_process:
        print("\nCalculating pruned model performance...")
    pruned_loss = calculate_test_l1_loss(net, test_iter, criterion)
    if is_main_process:
        loss_change = (pruned_loss - original_loss) / original_loss if original_loss > 0 else 0.0
        print(f"Pruned model L1 loss: {pruned_loss:.6f}")
        print(f"Loss change: {loss_change:.2%}")

    # 关键修改：将损失值传递给剪枝器
    pruner.set_loss_info(original_loss, pruned_loss)

    # 6. 保存剪枝模型（仅主进程）
    pruner.save_pruned_model(opt.save_pruned_model)

    # 可选：导出剪枝报告（仅主进程）
    if is_main_process:
        report_path = opt.save_pruned_model.replace('.pt', '_pruning_report.json')
        pruner.export_pruning_report(report_path)

    # 7. 测试剪枝后模型（可选，仅主进程）
    # if is_main_process:
    #     test(test_iter, net, criterion)

    # 8. 打印最终总结（仅主进程）
    if is_main_process:
        print("\n=== Final Threshold Pruning Summary ===")
        print(f"Global threshold: {opt.prune_threshold:.6f}")
        print(f"Layer-specific thresholds: {threshold_dict if threshold_dict else 'None'}")
        print(f"Prune target: {opt.prune_target}")
        print(f"Original loss: {original_loss:.6f}")
        print(f"Pruned loss: {pruned_loss:.6f}")
        print(f"Effective parameter reduction: {param_reduction_ratio:.2%}")
        print(f"Pruned model saved to: {opt.save_pruned_model}")
        print(f"Pruning report saved to: {report_path}")
