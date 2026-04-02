import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_dims):
        """
        初始化方式：net = MLP([输入维度, 隐藏层1, 隐藏层2, ..., 输出维度])
        例如：net = MLP([opt.input_nc, 524,524,524, opt.output_nc])
        """
        super(MLP, self).__init__()
        layers = []

        # 自动遍历维度列表，构建所有层
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]

            # 线性层
            layers.append(nn.Linear(in_dim, out_dim))

            # 不是最后一层 → 添加激活函数
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU(inplace=True))

        # 组合成网络
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
