from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import numpy as np
import h5py
import os

# worker 随机种子初始化
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# 判断主进程（只主进程打印日志）
def is_main_process():
    return int(os.environ.get('RANK', 0)) == 0

# numpy -> torch.Tensor
def numpy_to_tensor(np_array):
    return torch.from_numpy(np_array).float()


class H5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        super().__init__()
        self.h5_path = h5_path
        self.transform = transform
        self._h5_file = None

        # 初始化时只读一次样本总数，不常驻文件句柄
        with h5py.File(self.h5_path, "r") as f:
            self.num_samples = f["X"].shape[0]

    def _get_h5_file(self):
        """每个 worker 进程只打开一次 h5 文件"""
        if self._h5_file is None:
            # 只读模式 + 内存映射，提升随机读取速度
            self._h5_file = h5py.File(self.h5_path, "r", libver="latest")
        return self._h5_file

    def __getitem__(self, index):
        f = self._get_h5_file()
        # 按索引读取单条样本（h5 随机读取高效）
        x = f["X"][index]
        y = f["Y"][index]

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        # 原有维度扩充逻辑保留
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        return x, y

    def __len__(self):
        return self.num_samples

    def __del__(self):
        # 进程退出自动关闭句柄，防止文件占用
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None


def load_dataset(opt):
    data_root = os.path.abspath(opt.data_root)

    # 改用 .h5 文件
    train_h5 = os.path.join(data_root, "Train/train.h5")
    val_h5   = os.path.join(data_root, "Val/val.h5")
    test_h5  = os.path.join(data_root, "Test/test.h5")

    train_data = H5Dataset(train_h5, transform=numpy_to_tensor)
    val_data   = H5Dataset(val_h5,   transform=numpy_to_tensor)
    test_data  = H5Dataset(test_h5,  transform=numpy_to_tensor)

    # 分布式判断
    import torch.distributed as dist
    is_distributed = dist.is_available() and dist.is_initialized()

    train_sampler = val_sampler = test_sampler = None
    train_shuffle = True

    if is_distributed:
        train_sampler = DistributedSampler(train_data)
        val_sampler   = DistributedSampler(val_data,  shuffle=False)
        test_sampler  = DistributedSampler(test_data, shuffle=False)
        train_shuffle = False  # 分布式由 sampler 控制打乱

    # 合理设置 worker 数，避免过载
    cpu_cnt = os.cpu_count() or 4
    num_workers = min(cpu_cnt // 2, 12)
    if is_main_process():
        print(f"DataLoader num_workers: {num_workers}")

    # 统一加载参数
    loader_cfg = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=worker_init_fn
    )

    train_loader = DataLoader(
        train_data,
        batch_size=opt.batchSize,
        shuffle=train_shuffle,
        sampler=train_sampler,
        drop_last=True,
        **loader_cfg
    )

    val_loader = DataLoader(
        val_data,
        batch_size=opt.batchSize,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **loader_cfg
    )

    test_loader = DataLoader(
        test_data,
        batch_size=opt.batchSize,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
        **loader_cfg
    )

    return train_loader, val_loader, test_loader
