from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np
import h5py
import os
import traceback
from torch.cuda import current_device  # 引入显卡设备管理

# 为DataLoader添加worker初始化函数（可选，进一步确保安全性）
def worker_init_fn(worker_id):
    # 每个worker进程独立设置随机种子，避免数据加载顺序一致
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# 工具函数：判断主进程（仅主进程打印关键日志）
def is_main_process():
    return int(os.environ.get('RANK', 0)) == 0


def numpy_to_tensor(np_array):
    """转换为tensor并自动移动到当前进程的显卡"""
    return torch.from_numpy(np_array).float()  # 移除.to(device)之类的GPU转移操作


def load_dataset(opt):
    # 转换为绝对路径，确保所有进程访问一致
    data_root = os.path.abspath(opt.data_root)

    # 初始化HDF5数据集（显卡训练优化版）
    train_data = HDF5Dataset(
        inputPath=os.path.join(data_root, 'Train/Train_input.h5'),
        labelPath=os.path.join(data_root, 'Train/Train_output.h5'),
        transform=numpy_to_tensor
    )
    test_data = HDF5Dataset(
        inputPath=os.path.join(data_root, 'Test/Test_input.h5'),
        labelPath=os.path.join(data_root, 'Test/Test_output.h5'),
        transform=numpy_to_tensor
    )
    val_data = HDF5Dataset(
        inputPath=os.path.join(data_root, 'Val/Val_input.h5'),
        labelPath=os.path.join(data_root, 'Val/Val_output.h5'),
        transform=numpy_to_tensor
    )

    # 分布式采样器（确保每个显卡进程处理不同数据分片）
    train_sampler = DistributedSampler(train_data)
    test_sampler = DistributedSampler(test_data, shuffle=False)
    val_sampler = DistributedSampler(val_data, shuffle=False)

    # 关键优化：显卡训练的worker数量控制（避免CPU-GPU数据传输瓶颈）
    # 2进程×4worker=8总worker，平衡CPU预处理和GPU计算
    num_workers = min(os.cpu_count() // 2, 15)
    if is_main_process():
        print(f"显卡进程配置：每个进程{num_workers}个worker，总worker数：{num_workers * 2}")



    # 构建DataLoader（显卡训练优化参数）
    train_loader = DataLoader(
        dataset=train_data,
        sampler=train_sampler,
        batch_size=opt.batchSize,
        num_workers=num_workers,
        pin_memory=True,  # 显卡训练必须开启，加速CPU到GPU的数据传输
        persistent_workers=True,  # 保持worker存活，减少重复初始化
        drop_last=True,  # 丢弃不完整batch，避免多显卡同步卡死
        prefetch_factor=2,  # 提前加载2个batch到内存，隐藏IO延迟
        worker_init_fn=worker_init_fn  # 新增：worker初始化函数
    )
    test_loader = DataLoader(
        dataset=test_data,
        sampler=test_sampler,
        batch_size=opt.batchSize,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        worker_init_fn=worker_init_fn  # 新增：worker初始化函数
    )
    val_loader = DataLoader(
        dataset=val_data,
        sampler=val_sampler,
        batch_size=opt.batchSize,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        worker_init_fn=worker_init_fn  # 新增：worker初始化函数
    )

    return train_loader, test_loader, val_loader


class HDF5Dataset(Dataset):
    def __init__(self, inputPath, labelPath, transform=None):
        super(HDF5Dataset, self).__init__()
        self.inputPath = inputPath  # 仅保存文件路径，不保存h5py对象
        self.labelPath = labelPath
        self.transform = transform

        # 预计算样本数（仅主进程执行，避免多进程重复计算）
        with h5py.File(inputPath, 'r', swmr=True) as f:
            self.num_samples = len(f['data'])
        with h5py.File(labelPath, 'r', swmr=True) as f:
            assert len(f['data']) == self.num_samples, "输入和标签样本数不匹配"

    def __getitem__(self, index):
        # 关键：在每个__getitem__中动态打开文件（每个worker进程独立打开）
        # 1. 读取输入数据
        with h5py.File(self.inputPath, 'r', swmr=True) as input_file:
            pca_input = input_file['data'][index]

        # 2. 读取标签数据
        with h5py.File(self.labelPath, 'r', swmr=True) as label_file:
            pca_label = label_file['data'][index]

        # 3. 转换为tensor
        if self.transform is not None:
            pca_input = self.transform(pca_input)
            pca_label = self.transform(pca_label)

        # 4. 增加维度
        pca_input = pca_input.unsqueeze(0)
        pca_label = pca_label.unsqueeze(0)

        return pca_input, pca_label

    def __len__(self):
        return self.num_samples
