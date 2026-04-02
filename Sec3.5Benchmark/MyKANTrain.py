import efficient_kan.kan as E_kan
import efficient_kan.ImprovedKAN as I_kan
import efficient_kan.ComplexKan as C_kan
import MLP_Model as MLP
import Siren_Model as Siren
import Attention_Model as Attention
import ResFC_Model as ResFC
import torch
import torch.nn as nn
import MyKANnetLoader  # 适配HDF5和分布式的数据集加载器
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from torch.distributed import get_rank, is_initialized
from torch.cuda import empty_cache
import torch.multiprocessing as mp  # 引入多进程模块


# --------------------------
# 关键修复：设置多进程启动方式为spawn
# --------------------------
def set_multiprocessing_start_method():
    try:
        mp.set_start_method('spawn')  # 强制使用spawn启动方式（CUDA多进程必需）
    except RuntimeError:
        pass  # 已设置过则忽略


# --------------------------
# 工具函数：控制主进程行为
# --------------------------
def is_main_process():
    """判断是否为主进程（仅主进程执行日志、保存等操作）"""
    return get_rank() == 0 if is_initialized() else True


def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def test(net, test_iter, criterion, lr_scheduler, device):
    total, correct, test_loss = 0, 0, 0
    test_l1_loss = 0
    # 将模型设置为测试模式
    net.eval()

    with torch.no_grad():
        print("*************** test ***************")
        for X, y in test_iter:
            # y = np.expand_dims(y, axis=1)
            # y = torch.from_numpy(y)
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # 用模型估算输出的结果
            output = net(X)

            l1_loss = criterion(output, y)
            if hasattr(net, 'get_l1_regularization'):
                l1_reg = net.get_l1_regularization()
            else:
                l1_reg = sum(p.abs().sum() for p in net.parameters())
            total_loss = l1_loss + 1e-8 * l1_reg

            # total_loss = l1_loss

            test_loss += total_loss.item()
            test_l1_loss += l1_loss.item()
            total += 1

    # 计算得到损失函数的均值
    test_loss_mean = test_loss / total
    test_l1_loss_mean = test_l1_loss / total

    if lr_scheduler is not None:
        lr_scheduler.step(test_loss_mean)
    print("test_total_loss_mean: {:.6f}" \
          .format(test_loss_mean))
    print("test_l1_loss_mean: {:.6f}" \
          .format(test_l1_loss_mean))
    print("************************************\n")
    net.train()

    return test_loss_mean


def train(net, train_iter, criterion, optimizer, num_epochs, num_print,
          lr_scheduler=None, test_iter=None, device=None, Resume=None, start_epoch=None, SaveTrainLossPath=None,
          SaveValLossPath=None, SaveLastPara=None):
    if is_main_process():
        if Resume == False:
            print('从头开始训练开始训练')

        else:
            print('从断点开始训练开始训练')

    net.train()
    record_train = []
    record_test = []
    rank = get_rank() if is_initialized() else 0  # 当前进程编号

    # --------------------------
    # 新增：早停机制初始化
    # --------------------------
    best_val_loss = float('inf')  # 最佳验证损失，初始化为无穷大
    patience = 20  # 容忍连续无改进的轮数
    patience_counter = 0  # 连续无改进计数器
    early_stop = False  # 早停标志

    # 断点续训时恢复早停相关状态
    if Resume and is_main_process():
        checkpoint = torch.load(SaveLastPara, map_location=device)
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
            patience_counter = checkpoint['patience_counter']
            print(f"恢复早停状态：最佳验证损失={best_val_loss:.6f}, 连续无改进轮数={patience_counter}")

    for epoch in range(start_epoch, num_epochs):
        # 检查是否触发早停
        if early_stop:
            if is_main_process():
                print(f"早停触发！连续{patience}轮验证损失未下降，终止训练")
            break

        # 分布式训练：每个epoch打乱数据（确保多进程数据分片一致）
        if is_initialized():
            train_iter.sampler.set_epoch(epoch)

        # 仅主进程打印epoch信息
        if is_main_process():
            print(f"========== epoch: [{epoch + 1}/{num_epochs}] ==========")
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            start_time = time.time()

        total, train_loss = 0, 0  # 每个进程独立累计损失
        train_l1_loss = 0

        # 迭代训练数据
        for i, (X, y) in enumerate(train_iter):
            # 在train函数的epoch循环中，每个进程打印进度（区分rank）
            rank = get_rank() if is_initialized() else 0
            print(f"[Rank {rank}] Epoch {epoch + 1}, step {i + 1}/{len(train_iter)} completed")

            # 数据移动到当前进程的设备（非阻塞传输，加速GPU利用率）
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # 模型前向传播
            output = net(X)
            l1_loss = criterion(output, y)

            # 2. L1正则化损失
            if hasattr(net, 'get_l1_regularization'):
                l1_reg = net.get_l1_regularization()
            else:
                # 通用方法：遍历所有参数计算L1
                l1_reg = sum(p.abs().sum() for p in net.parameters())

            # 总损失 = L1损失 + λ * L1正则化
            total_loss = l1_loss + 1e-8 * l1_reg

            # total_loss = l1_loss

            # 反向传播与参数更新
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 累计损失（每个进程独立计算）
            train_loss += total_loss.item()
            train_l1_loss += l1_loss.item()
            total += 1
            train_loss_mean = train_loss / total
            train_l1_loss_mean = train_l1_loss / total

            # 仅主进程打印训练进度
            if is_main_process() and (i + 1) % num_print == 0:
                print(
                    f"step: [{i + 1}/{len(train_iter)}] | train_total_loss_mean: {train_loss_mean:.6f} | train_l1_loss_mean: {train_l1_loss_mean:.6f} | lr: {get_cur_lr(optimizer):.6f}")

        # 主进程记录训练损失并打印耗时
        if is_main_process():
            record_train.append(train_loss_mean)
            SaveLoss(SaveTrainLossPath, train_loss_mean)
            print(f"--- cost time: {time.time() - start_time:.4f}s ---\n")

            # 执行测试并记录验证损失
            val_loss = None
            if test_iter is not None:
                val_loss = test(net, test_iter, criterion, lr_scheduler, device)
                record_test.append(val_loss)
                SaveLoss(SaveValLossPath, val_loss)

                # --------------------------
                # 新增：早停逻辑判断
                # --------------------------
                # 验证损失下降（优化）
                if val_loss < best_val_loss - 1e-4:  # 加微小阈值避免浮点误差
                    best_val_loss = val_loss
                    patience_counter = 0  # 重置计数器
                    print(f"验证损失改善！更新最佳损失为 {best_val_loss:.6f}，计数器重置为0")
                else:
                    patience_counter += 1  # 计数器加1
                    print(f"验证损失未改善，连续无改进轮数: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        early_stop = True  # 触发早停

            # 保存最近模型（仅主进程，避免多进程写冲突）
            # torch.save(net.module.state_dict() if is_initialized() else net.state_dict(), opt.SaveLastPara)
            checkpoint = {'parameter': net.module.state_dict() if is_initialized() else net.state_dict(),
                          'scheduler': lr_scheduler.state_dict() if lr_scheduler else None,
                          'epoch': epoch,
                          # --------------------------
                          # 新增：保存早停相关状态
                          # --------------------------
                          'best_val_loss': best_val_loss,
                          'patience_counter': patience_counter}
            torch.save(checkpoint, SaveLastPara)
            print('已保存epoch:' + str(epoch + 1))

        # 每个epoch后清理GPU缓存，避免显存泄漏
        empty_cache()

    return record_train, record_test


def learning_curve(record_train, record_test=None):
    plt.style.use("ggplot")

    plt.plot(range(1, len(record_train) + 1), record_train, label="train_loss_mean")
    if record_test is not None:
        plt.plot(range(1, len(record_test) + 1), record_test, label="val_loss_mean")

    plt.legend(loc=4)
    plt.title("learning curve")
    plt.xticks(range(0, len(record_train) + 1, 5))
    plt.ylim(0, np.min(record_train) * 10)
    plt.xlabel("epoch")
    plt.ylabel("loss_mean")
    plt.ylim(ymin=0, ymax=1)

    plt.show()


def SaveLoss(Path, Lossmean):
    f = open(Path, "a")
    f.write(str(Lossmean) + "\n")

    f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Resume', type=bool, default=True, help='是否从中断的状态训练')

    parser.add_argument('--data_root', type=str,
                        default="/media/aiofm/F/20250723_k-fold-cross-validation-KAN/1-fold",
                        help='HDF5数据集根目录')
    parser.add_argument('--input_nc', type=int, default=33, help='输入特征维度')
    parser.add_argument('--output_nc', type=int, default=70, help='输出特征维度')
    parser.add_argument('--batchSize', type=int, default=5120,  # 关键：多显卡训练减小单进程batch_size
                        help='单进程批次大小（总批次=进程数×batchSize）')
    parser.add_argument('--learn_rate', type=float, default=0.00005, help='初始学习率')
    parser.add_argument('--num_epochs', type=int, default=300, help='训练轮数')

    # # 保存路径（确保主进程有写入权限）
    # parser.add_argument('--SaveTrainLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/record_train_loss_1_fold-524-524-524.txt")
    # parser.add_argument('--SaveValLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/record_val_loss_1_fold-524-524-524.txt")
    # parser.add_argument('--SavePara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/Kan_Para_1_fold-524-524-524.pt")
    # parser.add_argument('--SaveLastPara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/Last_Kan_Para_1_fold-524-524-524.pt")

    # # 保存路径（确保主进程有写入权限）
    # parser.add_argument('--SaveTrainLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/record_train_loss_1_fold-2375-2375-2375.txt")
    # parser.add_argument('--SaveValLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/record_val_loss_1_fold-2375-2375-2375.txt")
    # parser.add_argument('--SavePara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/Kan_Para_1_fold-2375-2375-2375.pt")
    # parser.add_argument('--SaveLastPara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16MLPParam/Last_Kan_Para_1_fold-2375-2375-2375.pt")

    # # 保存路径（确保主进程有写入权限）
    # parser.add_argument('--SaveTrainLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/record_train_loss_1_fold-1946-1946-1946-1946.txt")
    # parser.add_argument('--SaveValLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/record_val_loss_1_fold-1946-1946-1946-1946.txt")
    # parser.add_argument('--SavePara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/Kan_Para_1_fold-1946-1946-1946-1946.pt")
    # parser.add_argument('--SaveLastPara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/Last_Kan_Para_1_fold-1946-1946-1946-1946.pt")

    # # 保存路径（确保主进程有写入权限）
    # parser.add_argument('--SaveTrainLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/record_train_loss_1_fold-935x14.txt")
    # parser.add_argument('--SaveValLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/record_val_loss_1_fold-935x14.txt")
    # parser.add_argument('--SavePara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/Kan_Para_1_fold-935x14.pt")
    # parser.add_argument('--SaveLastPara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_ResFCParam/Last_Kan_Para_1_fold-935x14.pt")

    # # 保存路径（确保主进程有写入权限）
    # parser.add_argument('--SaveTrainLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/record_train_loss_1_fold-524-524-524.txt")
    # parser.add_argument('--SaveValLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/record_val_loss_1_fold-524-524-524.txt")
    # parser.add_argument('--SavePara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/Kan_Para_1_fold-524-524-524.pt")
    # parser.add_argument('--SaveLastPara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/Last_Kan_Para_1_fold-524-524-524.pt")

    # # 保存路径（确保主进程有写入权限）
    # parser.add_argument('--SaveTrainLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/record_train_loss_1_fold-2375-2375-2375.txt")
    # parser.add_argument('--SaveValLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/record_val_loss_1_fold-2375-2375-2375.txt")
    # parser.add_argument('--SavePara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/Kan_Para_1_fold-2375-2375-2375.pt")
    # parser.add_argument('--SaveLastPara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16SirenParam/Last_Kan_Para_1_fold-2375-2375-2375.pt")

    # # 保存路径（确保主进程有写入权限）
    # parser.add_argument('--SaveTrainLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/record_train_loss_1_fold-524-524-524.txt")
    # parser.add_argument('--SaveValLossPath', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/record_val_loss_1_fold-524-524-524.txt")
    # parser.add_argument('--SavePara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/Kan_Para_1_fold-524-524-524.pt")
    # parser.add_argument('--SaveLastPara', type=str,
    #                     default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/Last_Kan_Para_1_fold-524-524-524.pt")

    # 保存路径（确保主进程有写入权限）
    parser.add_argument('--SaveTrainLossPath', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/record_train_loss_1_fold-564-564-564.txt")
    parser.add_argument('--SaveValLossPath', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/record_val_loss_1_fold-564-564-564.txt")
    parser.add_argument('--SavePara', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/Kan_Para_1_fold-564-564-564.pt")
    parser.add_argument('--SaveLastPara', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16AttentionParam/Last_Kan_Para_1_fold-564-564-564.pt")


    parser.add_argument('--num_print', type=int, default=10, help='每num_print步打印一次损失')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    return parser.parse_args()


def main():
    global opt
    opt = parse_args()

    # 初始化分布式环境
    if opt.local_rank != -1:
        torch.cuda.set_device(opt.local_rank)  # 绑定当前进程到指定GPU
        device = torch.device("cuda", opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')  # 使用nccl后端（GPU推荐）
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 仅主进程打印设备信息
    if is_main_process():
        print(f"使用设备: {device}")
        print(f"分布式训练: {'启用' if is_initialized() else '禁用'}")
        print(f"总GPU数量: {torch.cuda.device_count()}")

    # net = MLP.MLP([opt.input_nc, 524, 524, 524, opt.output_nc])
    # net = MLP.MLP([opt.input_nc, 2375, 2375, 2375, opt.output_nc])
    # net = ResFC.ResFC(
    #     [opt.input_nc, 1946, 1946, 1946, 1946, opt.output_nc])
    # net = ResFC.ResFC(
    #     [opt.input_nc, 935, 935, 935, 935, 935, 935, 935, 935, 935, 935, 935, 935, 935, 935, opt.output_nc])
    # net = Siren.Siren([opt.input_nc, 524, 524, 524, opt.output_nc],omega_0=15)
    # net = Siren.Siren([opt.input_nc, 2375, 2375, 2375, opt.output_nc],omega_0=15)
    # net = Attention.Attention([opt.input_nc, 524, 524, 524, opt.output_nc], heads=4, dropout=0.0)
    net = Attention.Attention([opt.input_nc, 564, 564, 564, opt.output_nc], heads=4, dropout=0.0)

    print(net)
    # 总参数（原逻辑）
    total_param = sum(p.numel() for p in net.parameters())
    # 可训练参数（新增）
    trainable_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"模型总参数数量: {total_param}")
    print(f"可训练参数数量: {trainable_param}")

    # 将模型放入GPU中并初始化权重
    net.to(device)

    # 多显卡包装（分布式数据并行）
    if is_initialized():
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False  # 加速训练（确保所有参数均被使用）
        )

    # 仅主进程打印模型信息
    if is_main_process():
        print(net)
        print(f"模型总参数: {sum(p.numel() for p in net.parameters())}")

    # 返回的是三个数据加载器
    train_iter, test_iter, val_iter = MyKANnetLoader.load_dataset(opt)

    # 定义损失函数和优化器（移动到设备）
    criterion = torch.nn.L1Loss().to(device)

    optimizer = optim.Adam(
        net.parameters(),
        lr=opt.learn_rate
    )

    # 设置学习率衰减，每经过20轮，学习率衰减为原来的0.1倍
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        factor=0.1,
                                                        patience=5,
                                                        verbose=True)
    # 如果重来的话重新加载网络数据
    start_epoch = 0
    if opt.Resume == True:
        checkpoint = torch.load(opt.SaveLastPara, map_location=device)  # 加载断点
        net.module.load_state_dict(checkpoint['parameter'])  # 加载模型可学习参数
        if checkpoint['scheduler'] is not None:
            lr_scheduler.load_state_dict(checkpoint['scheduler'])  # 加载学习率参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    else:
        if is_main_process():
            # 删除之前创建的记录路径的txt文件
            if os.path.exists(opt.SaveTrainLossPath):  # 如果文件存在
                os.remove(opt.SaveTrainLossPath)
            if os.path.exists(opt.SaveValLossPath):  # 如果文件存在
                os.remove(opt.SaveValLossPath)
        start_epoch = 0

    # 开始训练
    record_train, record_val = train(
        net, train_iter, criterion, optimizer,
        opt.num_epochs, opt.num_print, lr_scheduler, val_iter, device, opt.Resume, start_epoch, opt.SaveTrainLossPath,
        opt.SaveValLossPath, opt.SaveLastPara
    )

    # 仅主进程执行后续操作（保存、绘图）
    if is_main_process():
        # 保存最终模型
        torch.save(
            net.module.state_dict() if is_initialized() else net.state_dict(),
            opt.SavePara
        )
        print(f"最终模型已保存至: {opt.SavePara}")
        # 绘制学习曲线
        learning_curve(record_train, record_val)


if __name__ == '__main__':
    # --------------------------
    # 关键修复：在程序入口设置spawn启动方式
    # --------------------------
    set_multiprocessing_start_method()

    torch.cuda.empty_cache()
    main()
