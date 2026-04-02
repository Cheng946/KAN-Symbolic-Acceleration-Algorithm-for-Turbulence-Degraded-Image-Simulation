import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.distributed import get_rank, is_initialized
from torch.cuda import empty_cache
import torch.multiprocessing as mp

# ========== 1. 初等函数配置（新增：集中管理支持的函数） ==========
# 定义所有支持的初等函数及其计算逻辑
SUPPORTED_ELEMENTARY_FUNCTIONS = {
    'silu': lambda x: torch.nn.functional.silu(x),
    'relu': lambda x: torch.nn.functional.relu(x),
    'sin': lambda x: torch.sin(x),
    'cos': lambda x: torch.cos(x),
    'exp': lambda x: torch.exp(torch.clamp(x, -10, 10)),  # 防止数值爆炸
    'log': lambda x: torch.log(torch.abs(x) + 1e-6),  # 防止log(0)
    'tanh': lambda x: torch.tanh(x),
    'sigmoid': lambda x: torch.sigmoid(x),
    'sqrt': lambda x: torch.sqrt(torch.abs(x) + 1e-6),  # 新增：平方根
    'square': lambda x: torch.square(x),  # 新增：平方
    'abs': lambda x: torch.abs(x),  # 新增：绝对值
    'identity': lambda x: x  # 新增：恒等函数
}

# 默认初等函数集合（可根据需求调整）
DEFAULT_ELEMENTARY_FUNCTIONS = ['silu', 'relu', 'tanh', 'sigmoid', 'abs', 'identity']


# ========== 2. SymbolicKAN 定义（核心修改：初等函数选择优化） ==========
class SymbolicKANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            scale_base=1.0,
            scale_mlp=1.0,
            base_activation=torch.nn.SiLU,
            elementary_functions=None,  # 修改：默认值改为None，便于后续处理
    ):
        super(SymbolicKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_base = scale_base
        self.scale_mlp = scale_mlp
        self.base_activation = base_activation()

        # 核心修改1：处理初等函数列表，验证有效性
        if elementary_functions is None:
            self.elementary_functions = DEFAULT_ELEMENTARY_FUNCTIONS
        else:
            # 验证所有指定的函数都受支持
            invalid_funcs = [f for f in elementary_functions if f not in SUPPORTED_ELEMENTARY_FUNCTIONS]
            if invalid_funcs:
                raise ValueError(
                    f"不支持的初等函数: {invalid_funcs}，支持的函数列表: {list(SUPPORTED_ELEMENTARY_FUNCTIONS.keys())}")
            self.elementary_functions = elementary_functions

        self.num_ef = len(self.elementary_functions)  # 初等函数数量

        # Base分支（冻结）
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        # 为每个初等函数创建独立的linear+bias
        self.ef_mlp_linears = nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(out_features, in_features))
            for _ in range(self.num_ef)
        ])
        self.ef_mlp_biases = nn.ParameterList([
            torch.nn.Parameter(torch.Tensor(out_features))
            for _ in range(self.num_ef)
        ])

        # 初等函数权重（训练）：每个函数的权重系数
        self.ef_weights = torch.nn.Parameter(torch.ones(self.num_ef))

        self.reset_parameters()

    def reset_parameters(self):
        import math
        # Base权重初始化（原有逻辑）
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        # 每个初等函数的MLP独立初始化
        for i in range(self.num_ef):
            # MLP linear初始化
            torch.nn.init.kaiming_uniform_(self.ef_mlp_linears[i], a=math.sqrt(5))
            # MLP bias初始化
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.ef_mlp_linears[i])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.ef_mlp_biases[i], -bound, bound)

        # 初等函数权重初始化
        torch.nn.init.constant_(self.ef_weights, 1.0 / self.num_ef)

    def apply_elementary_function(self, x, func_idx):
        """单个初等函数的前向计算（对应独立MLP）"""
        import torch.nn.functional as F
        func_name = self.elementary_functions[func_idx]
        # 使用当前函数对应的独立MLP计算
        mlp_output = F.linear(x, self.ef_mlp_linears[func_idx], self.ef_mlp_biases[func_idx])

        # 核心修改2：使用预定义的函数字典，简化逻辑
        ef_output = SUPPORTED_ELEMENTARY_FUNCTIONS[func_name](mlp_output)

        # 乘以当前函数的权重系数
        return ef_output * self.ef_weights[func_idx]

    def forward(self, x: torch.Tensor):
        import torch.nn.functional as F
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # Base分支前向（原有逻辑）
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # 每个初等函数独立MLP计算
        mlp_outputs = []
        for i in range(self.num_ef):
            # 每个函数用自己的MLP计算并加权
            ef_mlp_output = self.apply_elementary_function(x, i)
            mlp_outputs.append(ef_mlp_output)

        # 所有初等函数MLP输出求和
        mlp_output = torch.stack(mlp_outputs, dim=-1).sum(dim=-1) * self.scale_mlp

        # 总输出 = Base分支 + 所有初等函数MLP分支之和
        output = base_output + mlp_output
        return output.reshape(*original_shape[:-1], self.out_features)

    def get_l1_regularization(self):
        # L1正则：仅计算初等函数权重（保持原有逻辑）
        return self.ef_weights.abs().sum()


class SymbolicKAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            scale_base=1.0,
            scale_mlp=1.0,
            base_activation=torch.nn.SiLU,
            elementary_functions=None,  # 修改：默认值改为None
    ):
        super(SymbolicKAN, self).__init__()

        self.layers = torch.nn.ModuleList()
        # 处理初等函数列表（统一验证）
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


# ========== 3. 权重迁移函数（适配训练流程，无需修改） ==========
def load_pruned_kan_base_weights_to_symbolic_kan(
        pruned_kan_ckpt_path,
        symbolic_kan_model,
        device="cuda" if torch.cuda.is_available() else "cpu"
):
    kan_state_dict = torch.load(pruned_kan_ckpt_path, map_location=device)
    symbolic_kan_model.eval()

    with torch.no_grad():
        for layer_idx, sym_kan_layer in enumerate(symbolic_kan_model.layers):
            kan_base_weight_key = f"layers.{layer_idx}.base_weight"
            if kan_base_weight_key not in kan_state_dict:
                raise ValueError(f"未找到键: {kan_base_weight_key}")

            kan_base_weight = kan_state_dict[kan_base_weight_key].to(device)
            if kan_base_weight.shape != sym_kan_layer.base_weight.shape:
                raise ValueError(
                    f"第{layer_idx}层形状不匹配！KAN: {kan_base_weight.shape}, SymbolicKAN: {sym_kan_layer.base_weight.shape}")

            sym_kan_layer.base_weight.copy_(kan_base_weight)
            if is_main_process():
                print(f"迁移第{layer_idx}层Base权重，形状: {kan_base_weight.shape}")

    return symbolic_kan_model


# ========== 4. 冻结Base权重函数（适配新结构） ==========
def freeze_base_weights(symbolic_kan_model):
    """冻结Base分支权重，仅让每个初等函数的MLP和ef_weights可训练"""
    for layer in symbolic_kan_model.layers:
        # 冻结base_weight
        layer.base_weight.requires_grad = False

        # 确保每个初等函数的MLP参数可训练
        for i in range(len(layer.elementary_functions)):
            layer.ef_mlp_linears[i].requires_grad = True
            layer.ef_mlp_biases[i].requires_grad = True

        # 确保初等函数权重可训练
        layer.ef_weights.requires_grad = True

    # 打印冻结信息（仅主进程）
    if is_main_process():
        print("\n=== 权重冻结状态 ===")
        total_frozen = 0
        total_trainable = 0
        for name, param in symbolic_kan_model.named_parameters():
            param_size = param.numel()
            if param.requires_grad:
                total_trainable += param_size
                print(f"✅ 可训练: {name} | 形状: {param.shape} | 参数数: {param_size:,}")
            else:
                total_frozen += param_size
                print(f"❌ 冻结: {name} | 形状: {param.shape} | 参数数: {param_size:,}")
        print(f"\n总冻结参数: {total_frozen:,}")
        print(f"总可训练参数: {total_trainable:,}")

    return symbolic_kan_model


# ========== 5. 原有训练工具函数（完整保留，关键修改：打印l1_loss） ==========
def set_multiprocessing_start_method():
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass


def is_main_process():
    return get_rank() == 0 if is_initialized() else True


def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def test(net, test_iter, criterion, lr_scheduler, device):
    total, correct, test_loss = 0, 0, 0
    total_l1_loss = 0  # 新增：统计l1_loss总和
    net.eval()

    with torch.no_grad():
        print("*************** test ***************")
        for X, y in test_iter:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            output = net(X)

            l1_loss = criterion(output, y)
            if hasattr(net, 'get_l1_regularization'):
                l1_reg = net.get_l1_regularization()
            else:
                l1_reg = sum(p.abs().sum() for p in net.parameters())
            total_loss = l1_loss + 1e-7 * l1_reg

            test_loss += total_loss.item()
            total_l1_loss += l1_loss.item()  # 累加l1_loss
            total += 1

    test_loss_mean = test_loss / total
    test_l1_loss_mean = total_l1_loss / total  # 计算l1_loss均值

    # 修改：同时打印total_loss和l1_loss
    if lr_scheduler is not None:
        lr_scheduler.step(test_loss_mean)
    print(f"test_total_loss_mean: {test_loss_mean:.6f} | test_l1_loss_mean: {test_l1_loss_mean:.6f}")
    print("************************************\n")
    net.train()

    return test_loss_mean


def train(net, train_iter, criterion, optimizer, num_epochs, num_print,
          lr_scheduler=None, test_iter=None, device=None, Resume=None, start_epoch=None,
          SaveTrainLossPath=None, SaveValLossPath=None, SaveLastPara=None):
    if is_main_process():
        if Resume == False:
            print('从头开始训练开始训练')
        else:
            print('从断点开始训练开始训练')

    net.train()
    record_train = []
    record_test = []
    rank = get_rank() if is_initialized() else 0

    for epoch in range(start_epoch, num_epochs):
        if is_initialized():
            train_iter.sampler.set_epoch(epoch)

        if is_main_process():
            print(f"========== epoch: [{epoch + 1}/{num_epochs}] ==========")
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            start_time = time.time()

        total, train_loss, train_l1_loss = 0, 0, 0  # 新增：train_l1_loss统计

        for i, (X, y) in enumerate(train_iter):
            rank = get_rank() if is_initialized() else 0
            print(f"[Rank {rank}] Epoch {epoch + 1}, step {i + 1}/{len(train_iter)} completed")

            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            output = net(X)
            l1_loss = criterion(output, y)

            if hasattr(net, 'get_l1_regularization'):
                l1_reg = net.get_l1_regularization()
            else:
                l1_reg = sum(p.abs().sum() for p in net.parameters())

            total_loss = l1_loss + 1e-7 * l1_reg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            train_l1_loss += l1_loss.item()  # 累加l1_loss
            total += 1
            train_loss_mean = train_loss / total
            train_l1_loss_mean = train_l1_loss / total  # 计算l1_loss均值

            # 修改：打印时同时输出total_loss和l1_loss
            if is_main_process() and (i + 1) % num_print == 0:
                print(
                    f"step: [{i + 1}/{len(train_iter)}] | train_total_loss_mean: {train_loss_mean:.6f} | train_l1_loss_mean: {train_l1_loss_mean:.6f} | lr: {get_cur_lr(optimizer):.6f}")

        if is_main_process():
            record_train.append(train_loss_mean)
            SaveLoss(SaveTrainLossPath, train_loss_mean)
            # 新增：打印epoch级别的total_loss和l1_loss
            print(f"epoch {epoch+1} train_total_loss_mean: {train_loss_mean:.6f} | train_l1_loss_mean: {train_l1_loss_mean:.6f}")
            print(f"--- cost time: {time.time() - start_time:.4f}s ---\n")

            if test_iter is not None:
                val_loss = test(net, test_iter, criterion, lr_scheduler, device)
                record_test.append(val_loss)
                SaveLoss(SaveValLossPath, val_loss)

            checkpoint = {'parameter': net.module.state_dict() if is_initialized() else net.state_dict(),
                          'scheduler': lr_scheduler.state_dict(),
                          'epoch': epoch}
            torch.save(checkpoint, SaveLastPara)
            print('已保存epoch:' + str(epoch + 1))

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
    parser.add_argument('--Resume', type=bool, default=False, help='是否从中断的状态训练')
    parser.add_argument('--pruned_kan_ckpt', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_PruneParam/Pruned_Kan_Para_1_fold_4L_-524-524-524_15_2.pt",
                        help='剪枝后KAN权重路径')
    parser.add_argument('--elementary_functions', type=str, nargs='+',
                        default=DEFAULT_ELEMENTARY_FUNCTIONS,  # 修改：使用统一的默认值
                        help=f'初等函数列表，支持的函数: {list(SUPPORTED_ELEMENTARY_FUNCTIONS.keys())}')

    # 原有参数（完整保留）
    parser.add_argument('--data_root', type=str,
                        default="/media/aiofm/F/20250723_k-fold-cross-validation-KAN/1-fold",
                        help='HDF5数据集根目录')
    parser.add_argument('--input_nc', type=int, default=33, help='输入特征维度')
    parser.add_argument('--output_nc', type=int, default=70, help='输出特征维度')
    parser.add_argument('--batchSize', type=int, default=10240, help='单进程批次大小')
    parser.add_argument('--learn_rate', type=float, default=5e-3, help='初始学习率')
    parser.add_argument('--num_epochs', type=int, default=300, help='训练轮数')

    # 保存路径（使用你原有的路径）
    parser.add_argument('--SaveTrainLossPath', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/record_train_loss_1_fold_4L_-524-524-524_15_2.txt")
    parser.add_argument('--SaveValLossPath', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/record_val_loss_1_fold_4L_-524-524-524_15_2.txt")
    parser.add_argument('--SavePara', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/SymbolicKAN_Para_1_fold_4L_-524-524-524_15_2.pt")
    parser.add_argument('--SaveLastPara', type=str,
                        default="/home/aiofm/PycharmProjects/MyKANNet/15e-16_KAN_Symbolic_FinetuneParam/Last_SymbolicKAN_Para_1_fold_4L_-524-524-524_15_2.pt")

    parser.add_argument('--num_print', type=int, default=10, help='每num_print步打印一次损失')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    return parser.parse_args()


# ========== 6. 主训练逻辑（适配初等函数新配置） ==========
def main():
    global opt
    opt = parse_args()

    # 初始化分布式环境
    if opt.local_rank != -1:
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print(f"使用设备: {device}")
        print(f"分布式训练: {'启用' if is_initialized() else '禁用'}")
        print(f"总GPU数量: {torch.cuda.device_count()}")
        print(f"使用的初等函数: {opt.elementary_functions}")  # 新增：打印选择的初等函数

    # 初始化SymbolicKAN（新结构）
    layers_hidden = [opt.input_nc, 524, 524, 524, opt.output_nc]
    net = SymbolicKAN(
        layers_hidden=layers_hidden,
        elementary_functions=opt.elementary_functions
    ).to(device)

    # 迁移剪枝后的Base权重
    if is_main_process():
        print("\n=== 迁移剪枝后KAN的Base权重 ===")
    net = load_pruned_kan_base_weights_to_symbolic_kan(
        pruned_kan_ckpt_path=opt.pruned_kan_ckpt,
        symbolic_kan_model=net,
        device=device
    )

    # 冻结Base权重（适配新结构）
    if is_main_process():
        print("\n=== 冻结Base分支权重 ===")
    net = freeze_base_weights(net)

    # 多显卡包装
    if is_initialized():
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False  # 必须设为False（冻结的参数不会被更新）
        )

    if is_main_process():
        print(net)
        print(f"模型总参数: {sum(p.numel() for p in net.parameters())}")
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"可训练参数: {trainable_params:,}")

    # 加载数据集（原有逻辑）
    import MyKANnetLoader  # 你的数据集加载器
    train_iter, test_iter, val_iter = MyKANnetLoader.load_dataset(opt)

    # 定义损失函数和优化器（仅优化可训练参数）
    criterion = torch.nn.L1Loss().to(device)

    # 核心：仅传入可训练参数给优化器
    optimizer = optim.Adam(
        [p for p in net.parameters() if p.requires_grad],
        lr=opt.learn_rate
    )

    # 学习率调度器（原有逻辑）
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode='min',
                                                        factor=0.1,
                                                        patience=5,
                                                        verbose=True)

    # 断点续训（原有逻辑）
    if opt.Resume == True:
        checkpoint = torch.load(opt.SaveLastPara, map_location=device)
        if is_initialized():
            net.module.load_state_dict(checkpoint['parameter'])
        else:
            net.load_state_dict(checkpoint['parameter'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
    else:
        if is_main_process():
            if os.path.exists(opt.SaveTrainLossPath):
                os.remove(opt.SaveTrainLossPath)
            if os.path.exists(opt.SaveValLossPath):
                os.remove(opt.SaveValLossPath)
        start_epoch = 0

    # 开始训练（原有逻辑）
    record_train, record_val = train(
        net, train_iter, criterion, optimizer,
        opt.num_epochs, opt.num_print, lr_scheduler, val_iter, device,
        opt.Resume, start_epoch, opt.SaveTrainLossPath, opt.SaveValLossPath, opt.SaveLastPara
    )

    # 保存最终模型（原有逻辑）
    if is_main_process():
        torch.save(
            net.module.state_dict() if is_initialized() else net.state_dict(),
            opt.SavePara
        )
        print(f"最终SymbolicKAN模型已保存至: {opt.SavePara}")
        learning_curve(record_train, record_val)


if __name__ == '__main__':
    set_multiprocessing_start_method()
    torch.cuda.empty_cache()
    main()
