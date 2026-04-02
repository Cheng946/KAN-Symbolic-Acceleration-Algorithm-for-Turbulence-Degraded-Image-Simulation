import torch
import torch.nn.functional as F
import math
import torch.nn as nn


class KANLinear(torch.nn.Module):
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
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

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

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
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

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
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
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def get_l1_regularization(self):
        """计算当前层的L1正则化损失（所有参数的L1范数）"""
        l1_loss = 0.0
        l1_loss += self.base_weight.abs().sum()  # base_weight的L1
        l1_loss += self.spline_weight.abs().sum()  # spline_weight的L1
        if self.enable_standalone_scale_spline:
            l1_loss += self.spline_scaler.abs().sum()  # spline_scaler的L1
        return l1_loss


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )


    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)


        return x

    def get_l1_regularization(self):
        """计算整个KAN网络的L1正则化损失"""
        total_l1 = 0.0
        for layer in self.layers:
            total_l1 += layer.get_l1_regularization()
        return total_l1


class DenseBlock(nn.Module):
    def __init__(self,in_features,out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],):
        super(DenseBlock,self).__init__()
        self.KAN_fc=KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
        self.Base_Act=base_activation()

    def forward(self,x):
        # print(x.shape)
        x=torch.cat(x,dim=2)
        return self.Base_Act(self.KAN_fc(x))


class DenseKAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(DenseKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.KAN_fc1=KANLinear(
                    layers_hidden[0],
                    layers_hidden[1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
        self.db1=DenseBlock(layers_hidden[0]+layers_hidden[1],
                    layers_hidden[2],
                    grid_size=grid_size,
                    spline_order=spline_order,)

        self.db2=DenseBlock(layers_hidden[0]+layers_hidden[1]+layers_hidden[2],
                    layers_hidden[3],
                    grid_size=grid_size,
                    spline_order=spline_order,
        )

        self.db3=DenseBlock(layers_hidden[0]+layers_hidden[1]+layers_hidden[2]+layers_hidden[3],
                    layers_hidden[4],
                    grid_size=grid_size,
                    spline_order=spline_order,
        )

        self.KAN_fc2=KANLinear(layers_hidden[0]+layers_hidden[1]+layers_hidden[2]+layers_hidden[3]+layers_hidden[4],
                    layers_hidden[4]*2,
                    grid_size=grid_size,
                    spline_order=spline_order,)

        self.drpout=nn.Dropout(0.5)

        self.KAN_fc3=KANLinear(layers_hidden[4]*2,
                    layers_hidden[4],
                    grid_size=grid_size,
                    spline_order=spline_order,)

        self.Base_Act=base_activation()

    def forward(self, x: torch.Tensor, update_grid=False):
        x1=self.Base_Act(self.KAN_fc1(x))
        x2=self.db1([x,x1])
        x3=self.db2([x,x1,x2])
        x4=self.db3([x,x1,x2,x3])

        x_cat=torch.cat([x,x1,x2,x3,x4],dim=2)

        x=self.Base_Act(self.KAN_fc2(x_cat))
        x=self.drpout(x)
        x=self.KAN_fc3(x)

        return x


class AttentionLayer(nn.Module):
    def __init__(self,input_dim):
        super(AttentionLayer,self).__init__()
        self.attention=nn.Linear(input_dim,1)

    def forward(self,x):
        attention_weight=F.softmax(self.attention(x),dim=1)
        context_vector=torch.sum(attention_weight*x,dim=1)

        return context_vector

class KANWithAttention(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANWithAttention, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation=base_activation()

        self.KAN_fc1=KANLinear(
                    layers_hidden[0],
                    layers_hidden[1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
        self.attention1=AttentionLayer(layers_hidden[1])

        self.KAN_fc2=KANLinear(layers_hidden[1],
                    layers_hidden[2],
                    grid_size=grid_size,
                    spline_order=spline_order,)

        self.attention2 = AttentionLayer(layers_hidden[2])

        self.KAN_fc3 = KANLinear(layers_hidden[2],
                                 layers_hidden[3],
                                 grid_size=grid_size,
                                 spline_order=spline_order, )

        self.attention3 = AttentionLayer(layers_hidden[3])

        self.KAN_fc4 = KANLinear(layers_hidden[3],
                                 layers_hidden[4],
                                 grid_size=grid_size,
                                 spline_order=spline_order, )

        self.attention4 = AttentionLayer(layers_hidden[4])

        self.KAN_fc5 = KANLinear(layers_hidden[4],
                                 layers_hidden[5],
                                 grid_size=grid_size,
                                 spline_order=spline_order, )


    def forward(self, x: torch.Tensor, update_grid=False):
        # print(x.shape)
        x=self.base_activation(self.KAN_fc1(x))
        x=self.attention1(x)
        x = x.unsqueeze(dim=1)
        # print(x.shape)
        x=self.base_activation(self.KAN_fc2(x))
        x = self.attention2(x)
        x = x.unsqueeze(dim=1)
        x=self.base_activation(self.KAN_fc3(x))
        x = self.attention3(x)
        x = x.unsqueeze(dim=1)
        x=self.base_activation(self.KAN_fc4(x))
        x=self.attention4(x)
        x = x.unsqueeze(dim=1)
        x=self.KAN_fc5(x)

        # print(x.shape)

        return x

class FilM(nn.Module):
    def __init__(self,input_dim,modulation_dim,

                 ):
        super(FilM,self).__init__()

        self.gamma=nn.Linear(modulation_dim,input_dim)
        self.beta=nn.Linear(modulation_dim,input_dim)

    def forward(self,x,z):
        gamma=torch.tanh(self.gamma(z))
        beta=torch.tanh(self.beta(z))


        return gamma*x+beta

class KANWithFiLM(nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANWithFiLM, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()

        self.KAN_fc1=KANLinear(layers_hidden[0],
                                 layers_hidden[1],
                               scale_noise=scale_noise,
                               scale_base=scale_base,
                               scale_spline=scale_spline,
                               base_activation=torch.nn.SiLU,
                               grid_eps=grid_eps,
                               grid_range=grid_range,)

        self.film1=FilM(layers_hidden[1],layers_hidden[1])



        self.KAN_fc2=KANLinear(layers_hidden[1],
                                 layers_hidden[2],
                               scale_noise=scale_noise,
                               scale_base=scale_base,
                               scale_spline=scale_spline,
                               base_activation=torch.nn.SiLU,
                               grid_eps=grid_eps,
                               grid_range=grid_range,)

        self.film2=FilM(layers_hidden[2],layers_hidden[2])


        self.KAN_fc3=KANLinear(layers_hidden[2],
                                 layers_hidden[3],
                               scale_noise=scale_noise,
                               scale_base=scale_base,
                               scale_spline=scale_spline,
                               base_activation=torch.nn.SiLU,
                               grid_eps=grid_eps,
                               grid_range=grid_range,)

        self.film3 = FilM(layers_hidden[3], layers_hidden[3])

        self.KAN_fc4=KANLinear(layers_hidden[3],
                                 layers_hidden[4],
                               scale_noise=scale_noise,
                               scale_base=scale_base,
                               scale_spline=scale_spline,
                               base_activation=torch.nn.SiLU,
                               grid_eps=grid_eps,
                               grid_range=grid_range,)

        self.film4= FilM(layers_hidden[4], layers_hidden[4])

        self.KAN_fc5 = KANLinear(layers_hidden[4],
                                 layers_hidden[5],
                                 scale_noise=scale_noise,
                                 scale_base=scale_base,
                                 scale_spline=scale_spline,
                                 base_activation=torch.nn.SiLU,
                                 grid_eps=grid_eps,
                                 grid_range=grid_range, )

    def forward(self,x):
        x=self.base_activation(self.KAN_fc1(x))
        x=self.film1(x,x)

        x=self.base_activation(self.KAN_fc2(x))
        x=self.film2(x,x)

        x=self.base_activation(self.KAN_fc3(x))
        x=self.film3(x,x)

        x = self.base_activation(self.KAN_fc4(x))
        x = self.film4(x, x)

        x = self.KAN_fc5(x)

        return x


class ResKAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(ResKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()

        # 验证输入输出维度
        if layers_hidden[0] != 33:
            raise ValueError(f"输入层维度必须为33，实际为{layers_hidden[0]}")
        if layers_hidden[-1] != 70:
            raise ValueError(f"输出层维度必须为70，实际为{layers_hidden[-1]}")

        self.layers = torch.nn.ModuleList()


        # 创建所有KAN层
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )


    def forward(self, x: torch.Tensor, update_grid=False):
        x0=self.layers[0](x) #x0为128维
        x1=self.layers[1](x0) #x1为256维
        x2=self.layers[2](x1) #x2为384维
        x3=self.layers[3](x2) #x3为512维
        x4=self.layers[4](x3) #x4为640维

        x5=self.layers[5](x4)+x4 #x5为640维
        x5=self.layers[6](x5)+x3 #512维
        x5=self.layers[7](x5)+x2 #384维
        x5=self.layers[8](x5)+x1 #256维
        x5=self.layers[9](x5)+x0 #128维

        x5=self.layers[10](x5) #70维

        return x5

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class AutocorrelationLayer(nn.Module):
    """自相关运算层，用于处理序列数据的自相关特性"""

    def __init__(self, max_lag=5):
        super().__init__()
        self.max_lag = max_lag  # 最大时间滞后

    def forward(self, x):
        # x形状: (batch_size, seq_len, features) 或 (batch_size, features)
        # 如果是特征向量，先扩展为序列长度为1的序列
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, features)

        batch_size, seq_len, features = x.shape
        autocorr_features = []

        # 计算不同滞后的自相关
        for lag in range(self.max_lag + 1):
            if lag == 0:
                # 滞后为0时，自相关就是信号本身的平方
                autocorr = x ** 2
            else:
                # 计算滞后lag的自相关
                if lag >= seq_len:
                    # 如果滞后大于序列长度，用零填充
                    autocorr = torch.zeros_like(x)
                else:
                    # 自相关 = 信号与滞后信号的乘积
                    autocorr = x[:, lag:] * x[:, :-lag]
                    # 填充以保持长度一致
                    autocorr = F.pad(autocorr, (0, 0, lag, 0))

            autocorr_features.append(autocorr)

        # 合并所有滞后的自相关特征
        x = torch.cat(autocorr_features, dim=-1)
        # 如果原始输入是特征向量，还原形状
        if seq_len == 1:
            x = x.squeeze(1)

        return x


class AutocorrKAN(nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            autocorrelation_layers=None,  # 指定在哪些层后添加自相关层
            max_lag=5  # 自相关最大滞后
    ):
        super(AutocorrKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.max_lag = max_lag

        # 确定自相关层的位置，默认为最后一层前
        if autocorrelation_layers is None:
            autocorrelation_layers = [len(layers_hidden) - 2]

        self.layers = nn.ModuleList()
        self.autocorrelation_layers = nn.ModuleList()

        for i, (in_features, out_features) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            # 添加KAN线性层
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

            # 在指定位置添加自相关层
            if i in autocorrelation_layers:
                self.autocorrelation_layers.append(AutocorrelationLayer(max_lag))
            else:
                self.autocorrelation_layers.append(None)

    def forward(self, x: torch.Tensor, update_grid=False):
        for i, (layer, autocorr_layer) in enumerate(zip(self.layers, self.autocorrelation_layers)):
            if update_grid:
                layer.update_grid(x)

            x = layer(x)

            # 应用自相关层（如果存在）
            if autocorr_layer is not None:
                x = autocorr_layer(x)

        x = x.unsqueeze(1)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class FullyCustomizableFCN(nn.Module):
    def __init__(self, all_dims, activation=nn.ReLU(inplace=True)):
        """
        完全自定义所有节点维度的全连接网络

        参数:
            all_dims (list of int): 完整的网络节点维度序列，包含输入层、所有隐藏层和输出层
                例如: [input_dim, hidden1_dim, hidden2_dim, ..., output_dim]
            activation (nn.Module, 可选): 激活函数，默认为ReLU
        """
        super(FullyCustomizableFCN, self).__init__()

        # 校验输入的维度序列是否有效（至少需要输入和输出两层）
        if len(all_dims) < 2:
            raise ValueError("all_dims 必须包含至少2个元素（输入维度和输出维度）")

        self.all_dims = all_dims  # 保存完整维度序列
        layers = []

        # 遍历维度序列，构建每一层
        for i in range(len(all_dims) - 1):
            # 添加全连接层（从当前维度到下一个维度）
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            # 最后一层（输出层）不添加激活函数，其他层添加
            if i != len(all_dims) - 2:
                layers.append(activation)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class ResidualFCN(nn.Module):
    def __init__(self, all_dims, activation=nn.ReLU(inplace=True)):
        """
        带残差结构的全连接网络，在相同维度节点间建立残差连接

        参数:
            all_dims (list of int): 完整的维度序列，包含输入、隐藏层和输出维度
            activation (nn.Module, 可选): 激活函数，默认为ReLU
        """
        super(ResidualFCN, self).__init__()

        self.all_dims = all_dims

        self.activation=activation

        # 构建网络层
        self.layers = nn.ModuleList()
        for i in range(len(self.all_dims) - 1):
            # 构建基本层：线性变换 + 激活函数（输出层除外）
            layers = [nn.Linear(self.all_dims[i], self.all_dims[i + 1])]
            self.layers.append(nn.Sequential(*layers))


    def forward(self, x):
        #11L
        x0 = self.layers[0](x)
        x0 = self.activation(x0)
        x1 = self.layers[1](x0)
        x1 = self.activation(x1)
        x2 = self.layers[2](x1)
        x2 = self.activation(x2)
        x3 = self.layers[3](x2)
        x3 = self.activation(x3)
        x4 = self.layers[4](x3)
        x4 = self.activation(x4)

        x5 = self.layers[5](x4)+x4
        x5 = self.activation(x5)
        x6 = self.layers[6](x5) + x3
        x6 = self.activation(x6)
        x7 = self.layers[7](x6) + x2
        x7 = self.activation(x7)
        x8 = self.layers[8](x7) + x1
        x8 = self.activation(x8)
        x9 = self.layers[9](x8) + x0
        x9 = self.activation(x9)

        x10=self.layers[10](x9)

        #13L
        # x0 = self.layers[0](x)
        # x0 = self.activation(x0)
        # x1 = self.layers[1](x0)
        # x1 = self.activation(x1)
        # x2 = self.layers[2](x1)
        # x2 = self.activation(x2)
        # x3 = self.layers[3](x2)
        # x3 = self.activation(x3)
        # x4 = self.layers[4](x3)
        # x4 = self.activation(x4)
        # x5 = self.layers[5](x4)
        # x5 = self.activation(x5)
        #
        # x6 = self.layers[6](x5)+x5
        # x6 = self.activation(x6)
        # x7 = self.layers[7](x6) + x4
        # x7 = self.activation(x7)
        # x8 = self.layers[8](x7) + x3
        # x8 = self.activation(x8)
        # x9 = self.layers[9](x8) + x2
        # x9 = self.activation(x9)
        # x10 = self.layers[10](x9) + x1
        # x10 = self.activation(x10)
        # x11 = self.layers[11](x10) + x0
        # x11 = self.activation(x11)
        #
        # x12= self.layers[12](x11)

        return x10