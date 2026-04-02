import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    专用自注意力神经网络，用于33维→70维函数拟合
    调用方式：Attention([opt.input_nc, 524, 524, 524, opt.output_nc], heads=8, dropout=0.0)
    :param dims: 维度列表 [输入维度, 隐藏层1, 隐藏层2, 隐藏层3, 输出维度]
    :param heads: 多头注意力头数，必须能整除特征维度
    :param dropout: dropout概率，默认0.0
    """

    def __init__(self, dims, heads=8, dropout=0.0):
        super(Attention, self).__init__()
        # 解包维度：输入、3层隐藏层、输出
        self.input_dim, self.hid1, self.hid2, self.hid3, self.output_dim = dims
        self.heads = heads

        # 维度校验：注意力机制要求特征维度能被头数整除
        assert self.hid1 % heads == 0, f"隐藏层{self.hid1}不能被heads={heads}整除"
        assert self.hid2 % heads == 0, f"隐藏层{self.hid2}不能被heads={heads}整除"
        assert self.hid3 % heads == 0, f"隐藏层{self.hid3}不能被heads={heads}整除"

        # 1. 输入投影层：33维 → 第一层隐藏维度
        self.in_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hid1),
            nn.ReLU(inplace=True)
        )

        # 2. 核心：独立定义每一层，不使用Sequential包装注意力
        # 注意力层1
        self.attn1 = nn.MultiheadAttention(embed_dim=self.hid1, num_heads=heads, dropout=dropout, batch_first=True)
        self.linear1_1 = nn.Linear(self.hid1, self.hid1 * 4)
        self.linear1_2 = nn.Linear(self.hid1 * 4, self.hid1)
        self.dropout1 = nn.Dropout(dropout)

        # 注意力层2
        self.attn2 = nn.MultiheadAttention(embed_dim=self.hid2, num_heads=heads, dropout=dropout, batch_first=True)
        self.linear2_1 = nn.Linear(self.hid2, self.hid2 * 4)
        self.linear2_2 = nn.Linear(self.hid2 * 4, self.hid2)
        self.dropout2 = nn.Dropout(dropout)

        # 注意力层3
        self.attn3 = nn.MultiheadAttention(embed_dim=self.hid3, num_heads=heads, dropout=dropout, batch_first=True)
        self.linear3_1 = nn.Linear(self.hid3, self.hid3 * 4)
        self.linear3_2 = nn.Linear(self.hid3 * 4, self.hid3)
        self.dropout3 = nn.Dropout(dropout)

        # 3. 输出投影层：最后一层隐藏维度 → 70维
        self.out_proj = nn.Linear(self.hid3, self.output_dim)

    def forward(self, x):
        """前向传播：输入[B,33] → 输出[B,70]"""
        # 输入投影
        x = self.in_proj(x)  # [B, 33] → [B, hid1]

        # 增加序列维度（适配PyTorch多头注意力输入格式）
        # x = x.unsqueeze(1)  # [B, hid1] → [B, 1, hid1]

        # ===================== 注意力块1 =====================
        attn_out, _ = self.attn1(x, x, x)
        x = x + attn_out  # 残差连接
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear1_1(x)
        x = F.relu(x)
        x = self.linear1_2(x)
        x = self.dropout1(x)

        # ===================== 注意力块2 =====================
        attn_out, _ = self.attn2(x, x, x)
        x = x + attn_out
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear2_1(x)
        x = F.relu(x)
        x = self.linear2_2(x)
        x = self.dropout2(x)

        # ===================== 注意力块3 =====================
        attn_out, _ = self.attn3(x, x, x)
        x = x + attn_out
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear3_1(x)
        x = F.relu(x)
        x = self.linear3_2(x)
        x = self.dropout3(x)

        # 压缩序列维度+输出投影
        x = x.squeeze(1)  # [B, 1, hid3] → [B, hid3]
        x = self.out_proj(x)  # [B, hid3] → [B, 70]
        x = x.unsqueeze(1)

        return x
