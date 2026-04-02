import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    # 保持原有KANLinear实现（基础组件，负责非线性映射与线性组合）
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
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

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

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
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
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
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
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_output = F.linear(self.base_activation(x), self.base_weight)
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
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class WeightedLinearUpSampling(nn.Module):
    """加权线性组合升维模块：通过多层KANLinear实现带非线性的加权升维"""

    def __init__(self, in_dim, up_dims, base_activation=nn.SiLU):
        super().__init__()
        # up_dims为升维路径，从in_dim逐步提升维度
        self.layers = nn.ModuleList()
        prev_dim = in_dim
        for dim in up_dims:
            self.layers.append(
                KANLinear(
                    in_features=prev_dim,
                    out_features=dim,
                    base_activation=base_activation,
                    grid_size=6  # 增加网格密度提升升维拟合能力
                )
            )
            prev_dim = dim
        self.final_dim = prev_dim  # 升维后的最终维度

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AutoCorrelationModule(nn.Module):
    """自相关运算模块：计算特征间的自相关并与原特征融合"""

    def __init__(self, feature_dim, reduce_ratio=0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.reduce_ratio = reduce_ratio  # 自相关矩阵降维比例（避免维度爆炸）
        self.reduce_dim = int(feature_dim * reduce_ratio)

        # 压缩自相关矩阵的KAN层（将d×d矩阵压缩为向量）
        self.kan_compress = KANLinear(
            in_features=feature_dim * feature_dim,
            out_features=self.reduce_dim,
            grid_size=4  # 自相关特征平滑性较好，网格可稍小
        )
        # 融合自相关特征与原特征的KAN层
        self.kan_fuse = KANLinear(
            in_features=feature_dim + self.reduce_dim,
            out_features=feature_dim,  # 保持维度一致，便于残差连接
            grid_size=6
        )

    def forward(self, x):
        # x shape: (batch_size, num_samples, feature_dim)
        batch_size, num_samples, feature_dim = x.shape

        # 1. 计算自相关矩阵（特征间的相关性）
        auto_corr = []
        for i in range(num_samples):
            xi = x[:, i, :]  # (batch_size, feature_dim)
            corr_mat = xi.unsqueeze(-1) @ xi.unsqueeze(1)  # (batch_size, feature_dim, feature_dim)
            auto_corr.append(corr_mat)
        auto_corr = torch.stack(auto_corr, dim=1)  # (batch_size, num_samples, feature_dim, feature_dim)

        # 2. 压缩自相关矩阵为向量
        corr_flat = auto_corr.view(batch_size, num_samples, -1)  # 展平为(batch_size, num_samples, feature_dim^2)
        corr_compressed = self.kan_compress(corr_flat)  # (batch_size, num_samples, reduce_dim)

        # 3. 融合自相关特征与原特征（残差连接）
        x_fused = torch.cat([x, corr_compressed], dim=-1)  # (batch_size, num_samples, feature_dim + reduce_dim)
        x_fused = self.kan_fuse(x_fused)  # (batch_size, num_samples, feature_dim)
        return x + x_fused  # 残差连接增强特征保留


class DimensionReductionModule(nn.Module):
    """降维映射模块：通过KAN逐步降维，最终输出70维"""

    def __init__(self, in_dim, down_dims, base_activation=nn.SiLU):
        super().__init__()
        # down_dims为降维路径，最终维度需为70
        self.layers = nn.ModuleList()
        prev_dim = in_dim
        for dim in down_dims:
            self.layers.append(
                KANLinear(
                    in_features=prev_dim,
                    out_features=dim,
                    base_activation=base_activation,
                    grid_size=6
                )
            )
            prev_dim = dim
        self.out_dim = prev_dim  # 最终输出维度（需为70）

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KANforCompositeFunction(nn.Module):
    """用于拟合复合函数的改进KAN网络：输入33维，输出70维"""

    def __init__(
            self,
            base_activation=nn.SiLU,
            grid_size=6,
            spline_order=3
    ):
        super().__init__()
        # 固定输入维度为33，输出维度为70
        self.input_dim = 33
        self.output_dim = 70

        # 1. 加权线性组合升维模块：33 → 64 → 128 → 256（升维路径）
        self.upsampler = WeightedLinearUpSampling(
            in_dim=self.input_dim,
            up_dims=[64, 128, 256],  # 从33逐步升维至256
            base_activation=base_activation
        )
        up_final_dim = self.upsampler.final_dim  # 256

        # 2. 自相关运算模块（使用升维后的维度）
        self.auto_correlator = AutoCorrelationModule(
            feature_dim=up_final_dim,
            reduce_ratio=0.5  # 自相关矩阵压缩比例
        )

        # 3. 降维映射模块：256 → 128 → 70（最终降维至70）
        self.downsampler = DimensionReductionModule(
            in_dim=up_final_dim,
            down_dims=[128, self.output_dim],  # 从256逐步降维至70
            base_activation=base_activation
        )

        # 最终输出层已通过降维模块确保为70维，无需额外设置

    def forward(self, x, update_grid=False):
        # x shape: (batch_size, num_samples, 33) → 输入必须为33维
        assert x.size(-1) == self.input_dim, f"输入维度必须为{self.input_dim}，实际为{x.size(-1)}"

        # 1. 加权线性组合升维：33 → 256
        x = self.upsampler(x)  # (batch_size, num_samples, 256)

        # 2. 自相关运算（可选更新网格）
        if update_grid:
            # 为自相关模块中的KAN层更新网格
            self.auto_correlator.kan_compress.update_grid(x.reshape(-1, x.size(-1)))
            self.auto_correlator.kan_fuse.update_grid(
                torch.cat([x, x[..., :self.auto_correlator.reduce_dim]], dim=-1).reshape(
                    -1, x.size(-1) + self.auto_correlator.reduce_dim
                )
            )
        x = self.auto_correlator(x)  # (batch_size, num_samples, 256)

        # 3. 降维映射：256 → 70
        x = self.downsampler(x)  # (batch_size, num_samples, 70)

        return x  # 输出70维

    def regularization_loss(self):
        # 整合所有KAN层的正则化损失
        loss = 0.0
        # 升维模块正则化
        for layer in self.upsampler.layers:
            loss += layer.regularization_loss()
        # 自相关模块正则化
        loss += self.auto_correlator.kan_compress.regularization_loss()
        loss += self.auto_correlator.kan_fuse.regularization_loss()
        # 降维模块正则化
        for layer in self.downsampler.layers:
            loss += layer.regularization_loss()
        return loss
