import torch
import torch.nn as nn


# ---------------------------
# Siren 基础层（不变）
# ---------------------------
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                nn.init.uniform_(self.linear.weight, -1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                bound = torch.sqrt(torch.tensor(6.0 / self.linear.in_features)) / self.omega_0
                nn.init.uniform_(self.linear.weight, -bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


# ---------------------------
# ✅ 支持层列表 dims 的 Siren
# ---------------------------
class Siren(nn.Module):
    def __init__(self, dims, omega_0=30.0):
        """
        :param dims: 层维度列表，例如 [33, 2375, 2375, 2375, 70]
        """
        super().__init__()
        self.dims = dims
        self.omega_0 = omega_0

        layers = []
        num_layers = len(dims) - 1  # 总层数

        for i in range(num_layers):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            # 最后一层：线性输出（回归任务）
            if i == num_layers - 1:
                layers.append(nn.Linear(in_dim, out_dim))

            # 前面所有层：Siren 层
            else:
                is_first = (i == 0)
                layers.append(SirenLayer(in_dim, out_dim, is_first=is_first, omega_0=omega_0))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
