import torch
import torch.nn as nn
import torch.nn.functional as F


class ResFC(nn.Module):
    """
    残差全连接网络 (Residual Fully Connected Network)
    调用方式：ResFC([输入维度, 隐藏层1, 隐藏层2, ..., 输出维度])
    规则：隐藏层维度相同时自动添加残差连接，维度不同时无残差连接
    """

    def __init__(self, layers):
        super(ResFC, self).__init__()
        self.layers = layers  # 层维度列表，如 [input_nc, 388, 388, 388, 388, output_nc]
        self.num_layers = len(layers) - 1  # 实际网络层数

        # 构建网络层列表
        self.fc_layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = layers[i]
            out_dim = layers[i + 1]
            self.fc_layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        """前向传播：核心残差逻辑"""
        out = x

        for i in range(self.num_layers):
            identity = out  # 保存残差分支输入
            out = self.fc_layers[i](out)  # 主分支线性变换

            # 非最后一层：激活函数 + 残差连接
            if i != self.num_layers - 1:
                out = F.relu(out)
                # 维度相同时添加残差连接（核心设计）
                if identity.shape == out.shape:
                    out = out + identity

        return out



