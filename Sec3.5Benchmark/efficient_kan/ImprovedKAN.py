import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseKANLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            sparsity=0.5,  # 稀疏度控制，0.5表示约50%的连接会被保留
            dynamic_sparsity=True,  # 是否动态调整稀疏连接
    ):
        super(SparseKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.sparsity = sparsity
        self.dynamic_sparsity = dynamic_sparsity

        # 计算网格
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # 基础权重和样条权重
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))

        # 创建可学习的稀疏掩码
        self.mask = nn.Parameter(
            torch.bernoulli(torch.full((out_features, in_features), 1 - sparsity)),
            requires_grad=dynamic_sparsity  # 如果是动态稀疏，允许掩码学习
        )

        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        # 应用掩码到基础权重初始化
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        self.base_weight.data *= self.mask

        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            # 应用掩码到样条权重
            self.spline_weight.data *= self.mask.unsqueeze(-1)

            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
                # 应用掩码到样条缩放器
                self.spline_scaler.data *= self.mask

    def b_splines(self, x: torch.Tensor):
        """计算B样条基函数"""
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8)  # 防止除零
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)] + 1e-8)  # 防止除零
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """计算插值曲线的系数"""
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        # 使用带正则化的最小二乘，提高稳定性
        solution = torch.linalg.lstsq(A, B, rcond=1e-3).solution
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        # 应用掩码到样条权重
        weighted = self.spline_weight * self.mask.unsqueeze(-1)
        if self.enable_standalone_scale_spline:
            weighted = weighted * self.spline_scaler.unsqueeze(-1)
        return weighted

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # 应用掩码到基础输出
        masked_base_weight = self.base_weight * self.mask
        base_output = F.linear(self.base_activation(x), masked_base_weight)

        # 样条输出
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)

        # 为每个特征单独计算网格，适应不同特征的分布
        grid_list = []
        for i in range(self.in_features):
            x_feature = x[:, i:i + 1]
            x_sorted = torch.sort(x_feature, dim=0)[0]

            # 确保有足够的数据点来计算网格
            if batch <= self.grid_size + 1:
                # 数据点不足时使用均匀网格
                grid_adaptive = torch.linspace(
                    x_sorted.min(), x_sorted.max(),
                    self.grid_size + 1, device=x.device
                ).unsqueeze(1)
            else:
                grid_adaptive = x_sorted[
                    torch.linspace(
                        0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
                    )
                ]

            uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
            grid_uniform = (
                    torch.arange(
                        self.grid_size + 1, dtype=torch.float32, device=x.device
                    ).unsqueeze(1)
                    * uniform_step
                    + x_sorted[0]
                    - margin
            )

            # 混合自适应网格和均匀网格
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            grid_list.append(grid)

        # 合并所有特征的网格
        grid = torch.cat(grid_list, dim=1).T

        # 添加边界点
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        new_coeff = self.curve2coeff(x, unreduced_spline_output)
        # 应用掩码到新系数
        self.spline_weight.data.copy_(new_coeff * self.mask.unsqueeze(-1))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0, regularize_sparsity=1.0):
        # 原始的正则化项
        l1_fake = self.spline_weight.abs().mean(-1) * self.mask
        regularization_loss_activation = l1_fake.sum()

        # 熵正则化
        p = l1_fake / (regularization_loss_activation + 1e-8)  # 防止除零
        regularization_loss_entropy = -torch.sum(p * (p + 1e-8).log())

        # 稀疏性正则化，鼓励更多连接被关闭
        if self.dynamic_sparsity:
            # 使用L1正则化鼓励掩码值接近0
            regularization_loss_sparsity = self.mask.sum()
        else:
            regularization_loss_sparsity = 0

        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
                + regularize_sparsity * regularization_loss_sparsity
        )


class ModularKAN(nn.Module):
    def __init__(
            self,
            in_features=33,
            out_features=70,
            num_modules=3,  # 将输入分成3个模块处理
            hidden_dim=24,  # 每个模块的隐藏维度
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            sparsity=0.6,  # 稀疏度设置，保留60%的连接
    ):
        super(ModularKAN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_modules = num_modules

        # 计算每个模块处理的输入特征数
        features_per_module = in_features // num_modules
        remaining_features = in_features % num_modules

        # 创建模块列表
        self.modules_list = nn.ModuleList()
        start = 0

        for i in range(num_modules):
            # 分配特征，最后一个模块处理剩余的特征
            end = start + features_per_module + (1 if i == num_modules - 1 else 0)

            # 每个模块从部分输入特征映射到隐藏维度
            self.modules_list.append(
                SparseKANLinear(
                    end - start,
                    hidden_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    sparsity=sparsity,
                )
            )
            start = end

        # 交叉模块连接，允许模块间信息交换
        self.cross_connections = nn.ModuleList()
        for i in range(num_modules):
            self.cross_connections.append(
                SparseKANLinear(
                    hidden_dim * num_modules,
                    hidden_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    sparsity=sparsity * 1.2,  # 交叉连接更稀疏
                )
            )

        # 最终输出层
        self.output_layer = SparseKANLinear(
            hidden_dim * num_modules,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
            sparsity=sparsity,
        )

        # 用于调整网格的激活函数
        self.activation = base_activation()

    def forward(self, x: torch.Tensor, update_grid=False):
        # 将输入分割到各个模块
        features_per_module = self.in_features // self.num_modules
        remaining_features = self.in_features % self.num_modules

        module_outputs = []
        start = 0

        for i in range(self.num_modules):
            end = start + features_per_module + (1 if i == self.num_modules - 1 else 0)
            x_module = x[..., start:end]

            if update_grid:
                self.modules_list[i].update_grid(x_module.reshape(-1, x_module.size(-1)))

            module_out = self.modules_list[i](x_module)
            module_outputs.append(module_out)
            start = end

        # 合并模块输出
        combined = torch.cat(module_outputs, dim=-1)

        # 交叉模块处理，允许信息在模块间流动
        cross_outputs = []
        for i in range(self.num_modules):
            if update_grid:
                self.cross_connections[i].update_grid(combined.reshape(-1, combined.size(-1)))
            cross_out = self.cross_connections[i](combined)
            cross_outputs.append(cross_out)

        # 再次合并
        cross_combined = torch.cat(cross_outputs, dim=-1)

        # 最终输出
        if update_grid:
            self.output_layer.update_grid(cross_combined.reshape(-1, cross_combined.size(-1)))

        output = self.output_layer(cross_combined)
        return output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0, regularize_sparsity=1.0):
        # 计算所有模块的正则化损失
        loss = sum(
            module.regularization_loss(regularize_activation, regularize_entropy, regularize_sparsity)
            for module in self.modules_list
        )
        # 加上交叉连接的正则化损失
        loss += sum(
            conn.regularization_loss(regularize_activation, regularize_entropy, regularize_sparsity)
            for conn in self.cross_connections
        )
        # 加上输出层的正则化损失
        loss += self.output_layer.regularization_loss(regularize_activation, regularize_entropy, regularize_sparsity)
        return loss


# 使用示例：创建一个从33维映射到70维的模型
def create_33to70_kan(**kwargs):
    return ModularKAN(
        in_features=33,
        out_features=70,
        num_modules=3,  # 33维分为3个模块，分别处理11、11、11维
        hidden_dim=24,
        grid_size=6,
        spline_order=3,
        sparsity=0.6, **kwargs
    )
