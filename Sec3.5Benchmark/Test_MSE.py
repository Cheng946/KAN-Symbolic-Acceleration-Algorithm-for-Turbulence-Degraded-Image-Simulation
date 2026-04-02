import efficient_kan.kan as E_kan
import MLP_Model as MLP
import ResFC_Model as ResFC
import Siren_Model as Siren
import Attention_Model as Attention
from SymbolicKAN_Finetune import SymbolicKAN,DEFAULT_ELEMENTARY_FUNCTIONS

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def zernikeGen(N, coeff, ZernPoly36, **kwargs):
    # Generating the Zernike Phase representation.

    num_coeff = coeff.shape[1]
    # print(num_coeff)

    # Setting up 2D grid
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), np.linspace(-1, 1, N, endpoint=True))

    zern_out = np.zeros((N, N, coeff.shape[1]))

    # 获取当前工作目录
    for i in range(0, num_coeff):
        zern_out[:, :, i] = coeff[0, i] * ZernPoly36[:, :, i + 3];

    return zern_out


def ComputeOTF(a):
    ZernPoly36 = np.load('/home/aiofm/PycharmProjects/MyKANNet/36—128ZernPoly.npy');

    # 采样网格数
    N = 128;

    zernike_stack = zernikeGen(N, a, ZernPoly36);
    # 计算相位
    Fai = np.sum(zernike_stack, axis=2);

    # 孔径函数
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), np.linspace(-1, 1, N, endpoint=True))
    mask = np.sqrt(x_grid ** 2 + y_grid ** 2) <= 1

    # 根据傅里叶变换得到对偶定理，F(F(w()))计算OTF
    wave = mask * np.exp(1j * 2 * np.pi * Fai);
    # wave进行归一化,离散Parseval定理np.sum(PSF)/(128**2)=np.sum(np.abs(wave)**2)
    p = np.sum(np.abs(wave) ** 2)
    wave = wave * (((1 / 128 ** 2) / p) ** 0.5)

    cor = correlate(wave, wave, mode='same') * N ** 2
    cor = cor[::-1, ::-1]

    return cor


parser = argparse.ArgumentParser()

parser.add_argument('--data_root', type=str, default="/media/aiofm/F/20250723_k-fold-cross-validation-KAN/1-fold",
                    help='数据集的根目录')
parser.add_argument('--input_nc', type=int, default=33, help='输入维度的通道数量')
parser.add_argument('--output_nc', type=int, default=70, help='输出维度的通道数量')
parser.add_argument('--batchSize', type=int, default=64, help='一次训练载入的数据量')
parser.add_argument('--learn_rate', type=float, default=0.00005, help='初始学习率')
parser.add_argument('--num_epochs', type=int, default=300, help='训练的轮数')

# # 保存路径（确保主进程有写入权限）
# parser.add_argument('--SaveTrainLossPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/record_train_loss_1_fold-524-524-524.txt")
# parser.add_argument('--SaveValLossPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/record_val_loss_1_fold-524-524-524.txt")
# parser.add_argument('--SavePara', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/Kan_Para_1_fold-524-524-524.pt")
# parser.add_argument('--SaveLastPara', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/Last_Kan_Para_1_fold-524-524-524.pt")
# # 新增：测试指标保存路径
# parser.add_argument('--SaveMetricsPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/test_metrics_1_fold-524-524-524.txt",
#                     help='测试集指标保存路径')

# # 保存路径（确保主进程有写入权限）
# parser.add_argument('--SaveTrainLossPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/record_train_loss_1_fold-2375-2375-2375.txt")
# parser.add_argument('--SaveValLossPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/record_val_loss_1_fold-2375-2375-2375.txt")
# parser.add_argument('--SavePara', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/Kan_Para_1_fold-2375-2375-2375.pt")
# parser.add_argument('--SaveLastPara', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/Last_Kan_Para_1_fold-2375-2375-2375.pt")
# # 新增：测试指标保存路径
# parser.add_argument('--SaveMetricsPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/test_metrics_1_fold-2375-2375-2375.txt",
#                     help='测试集指标保存路径')

# # 保存路径（确保主进程有写入权限）
# parser.add_argument('--SaveTrainLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/record_train_loss_1_fold-1946-1946-1946-1946.txt")
# parser.add_argument('--SaveValLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/record_val_loss_1_fold-1946-1946-1946-1946.txt")
# parser.add_argument('--SavePara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/Kan_Para_1_fold-1946-1946-1946-1946.pt")
# parser.add_argument('--SaveLastPara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/Last_Kan_Para_1_fold-1946-1946-1946-1946.pt")
# parser.add_argument('--SaveMetricsPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/test_metrics_1_fold-1946-1946-1946-1946.txt",
#                     help='测试集指标保存路径')

# # 保存路径（确保主进程有写入权限）
# parser.add_argument('--SaveTrainLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/record_train_loss_1_fold-935x14.txt")
# parser.add_argument('--SaveValLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/record_val_loss_1_fold-935x14.txt")
# parser.add_argument('--SavePara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/Kan_Para_1_fold-935x14.pt")
# parser.add_argument('--SaveLastPara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/Last_Kan_Para_1_fold-935x14.pt")
# parser.add_argument('--SaveMetricsPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/test_metrics_1_fold-935x14.txt",
#                     help='测试集指标保存路径')

# # 保存路径（确保主进程有写入权限）
# parser.add_argument('--SaveTrainLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/record_train_loss_1_fold-524-524-524.txt")
# parser.add_argument('--SaveValLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/record_val_loss_1_fold-524-524-524.txt")
# parser.add_argument('--SavePara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/Kan_Para_1_fold-524-524-524.pt")
# parser.add_argument('--SaveLastPara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/Last_Kan_Para_1_fold-524-524-524.pt")
# parser.add_argument('--SaveMetricsPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/test_metrics_1_fold-524-524-524.txt",
#                     help='测试集指标保存路径')

# # 保存路径（确保主进程有写入权限）
# parser.add_argument('--SaveTrainLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/record_train_loss_1_fold-2375-2375-2375.txt")
# parser.add_argument('--SaveValLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/record_val_loss_1_fold-2375-2375-2375.txt")
# parser.add_argument('--SavePara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/Kan_Para_1_fold-2375-2375-2375.pt")
# parser.add_argument('--SaveLastPara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/Last_Kan_Para_1_fold-2375-2375-2375.pt")
# parser.add_argument('--SaveMetricsPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/test_metrics_1_fold-2375-2375-2375.txt",
#                     help='测试集指标保存路径')

# # 保存路径（确保主进程有写入权限）
# parser.add_argument('--SaveTrainLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/record_train_loss_1_fold-524-524-524.txt")
# parser.add_argument('--SaveValLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/record_val_loss_1_fold-524-524-524.txt")
# parser.add_argument('--SavePara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/Kan_Para_1_fold-524-524-524.pt")
# parser.add_argument('--SaveLastPara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/Last_Kan_Para_1_fold-524-524-524.pt")
# parser.add_argument('--SaveMetricsPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/test_metrics_1_fold-524-524-524.txt",
#                     help=='测试集指标保存路径')

# # 保存路径（确保主进程有写入权限）
# parser.add_argument('--SaveTrainLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/record_train_loss_1_fold-564-564-564.txt")
# parser.add_argument('--SaveValLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/record_val_loss_1_fold-564-564-564.txt")
# parser.add_argument('--SavePara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/Kan_Para_1_fold-564-564-564.pt")
# parser.add_argument('--SaveLastPara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/Last_Kan_Para_1_fold-564-564-564.pt")
# parser.add_argument('--SaveMetricsPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/test_metrics_1_fold-564-564-564.txt")

# # 保存路径（确保主进程有写入权限）
# parser.add_argument('--SaveTrainLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN4LayerParam/record_train_loss_1_fold-524-524-524_15_2.txt")
# parser.add_argument('--SaveValLossPath', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN4LayerParam/record_val_loss_1_fold-524-524-524_15_2.txt")
# parser.add_argument('--SavePara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN4LayerParam/Kan_Para_1_fold-524-524-524_15_2.pt")
# parser.add_argument('--SaveLastPara', type=str,
#                         default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN4LayerParam/Last_Kan_Para_1_fold-524-524-524_15_2.pt")
# parser.add_argument('--SaveMetricsPath', type=str,
#                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN4LayerParam/test_metrics_1_fold-524-524-524_15_2.txt",
#                     help='测试集指标保存路径')

# 保存路径（确保主进程有写入权限）
parser.add_argument('--elementary_functions', type=str, nargs='+', default=DEFAULT_ELEMENTARY_FUNCTIONS)
parser.add_argument('--SaveLastPara', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/Last_SymbolicKAN_Para_1_fold_4L_-524-524-524_15_2.pt")
parser.add_argument('--SaveMetricsPath', type=str,
                    default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/test_metrics_1_fold-524-524-524_15_2.txt",
                    help='测试集指标保存路径')

parser.add_argument('--num_print', type=int, default=10, help='每过num_print轮训练打印一次损失')
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

opt = parser.parse_args()

if opt.local_rank != -1:
    torch.cuda.set_device(opt.local_rank)
    device = torch.device("cuda", opt.local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')


# 测试网络（可视化）
def test(test_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, label = data
            inputs, label = inputs.to(device), label.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            print(f"推理时间: {end_time - start_time:.4f} s")

            # 可视化对比
            outputs1 = outputs[20].cpu().numpy()
            label1 = label[20].cpu().numpy()

            plt.plot(range(outputs1.shape[1]), outputs1[0, :], color='blue', marker='o', linestyle='-', label='output')
            plt.plot(range(label1.shape[1]), label1[0, :], color='red', marker='x', linestyle='-', label='label')
            plt.ylim(-2.5, 2.5)
            plt.legend()
            plt.show()
            break
        print("可视化测试完成")


# ===================== 全指标计算：MSE, MAE, MAPE, R² =====================
def calculate_all_metrics(model, test_loader, device):
    """
    计算测试集全指标：MSE, MAE, MAPE, R²
    适配你的KAN+OTF回归任务
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)

            # 展平用于全局指标计算
            preds_np = preds.cpu().numpy().reshape(-1)
            labels_np = targets.cpu().numpy().reshape(-1)

            all_preds.append(preds_np)
            all_labels.append(labels_np)

    # 合并整个测试集
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    epsilon = 1e-8  # 防止除0

    # 1. MSE
    mse = mean_squared_error(all_labels, all_preds)

    # 2. MAE
    mae = mean_absolute_error(all_labels, all_preds)

    # 3. MAPE (%)
    mape = np.mean(np.abs((all_labels - all_preds) / (all_labels + epsilon))) * 100

    # 4. R²
    r2 = r2_score(all_labels, all_preds)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'MAPE(%)': mape,
        'R²': r2
    }

    return metrics


# 保存指标到文件
def save_metrics(metrics, save_path):
    """
    将测试指标保存到指定txt文件
    """
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("           测试集指标结果\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k:>10}: {v:.6f}\n")
        f.write("=" * 50 + "\n")
    print(f"指标已保存到：{save_path}")


# 原L1损失函数
def calculate_test_l1_loss(model, test_loader, criterion):
    model.eval()
    model.to(device)
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


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # 构建模型
    # net = MLP.MLP([opt.input_nc, 524, 524, 524, opt.output_nc])
    # net = MLP.MLP([opt.input_nc, 2375, 2375, 2375, opt.output_nc])
    # net = ResFC.ResFC([opt.input_nc, 1946, 1946, 1946, 1946, opt.output_nc])
    # net = ResFC.ResFC(
    #     [opt.input_nc, 935, 935, 935, 935, 935, 935, 935, 935, 935, 935, 935, 935, 935, 935, opt.output_nc])
    # net = Siren.Siren([opt.input_nc, 524, 524, 524, opt.output_nc], omega_0=15)
    # net = Siren.Siren([opt.input_nc, 2375, 2375, 2375, opt.output_nc], omega_0=15)
    # net = Attention.Attention([opt.input_nc, 524, 524, 524, opt.output_nc], heads=4, dropout=0.0)
    # net = Attention.Attention([opt.input_nc, 564, 564, 564, opt.output_nc], heads=4, dropout=0.0)
    # net = E_kan.KAN([opt.input_nc, 524, 524, 524, opt.output_nc], grid_size=15, spline_order=2)
    net = SymbolicKAN(layers_hidden=[opt.input_nc, 524, 524, 524, opt.output_nc], elementary_functions=opt.elementary_functions)

    # 总参数（原逻辑）
    total_param = sum(p.numel() for p in net.parameters())
    # 可训练参数（新增）
    trainable_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"模型总参数数量: {total_param}")
    print(f"可训练参数数量: {trainable_param}")

    net = net.to(device)

    # 多卡并行
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('use {} gpus!'.format(num_gpus))
        net = nn.parallel.DistributedDataParallel(net, device_ids=[opt.local_rank], output_device=opt.local_rank)

    # 加载训练好的权重
    # net.module.load_state_dict(torch.load(opt.SavePara))

    checkpoint = torch.load(opt.SaveLastPara)
    net.module.load_state_dict(checkpoint['parameter'])


    # 加载数据
    train_iter, test_iter, val_test = MyKANnetLoader.load_dataset(opt)
    criterion = torch.nn.L1Loss()

    # ===================== 运行全指标测试 =====================
    print("\n" + "=" * 60)
    print("           开始计算测试集全指标")
    print("=" * 60)

    metrics = calculate_all_metrics(net, test_iter, device)

    # 打印结果
    for k, v in metrics.items():
        print(f"{k:>10}: {v:.6f}")

    # 保存指标到指定路径
    save_metrics(metrics, opt.SaveMetricsPath)

    print("=" * 60 + "\n")

    # 可视化测试（保留原有功能）
    # test(test_iter, net, criterion)
