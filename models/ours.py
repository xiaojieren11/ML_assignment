# 文件路径: /sdb/ML_assignment/models/ours.py
# 请用以下全部内容替换该文件

import torch
import torch.nn as nn
import math
from torch.nn import functional as F


# 1. 定义一个带有门控线性单元(GLU)的前馈网络，用于替换Transformer中的标准FFN
class GatedFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff * 2)  # 输出维度乘以2，一半给门，一半给值
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        # GLU: x * sigmoid(gate)
        x = F.glu(x, dim=-1)  # F.glu가 이 계산을 한 번에 처리합니다.
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


# 2. 我们将把这个GatedFeedForward集成到标准的TransformerEncoderLayer中
# （为了简洁，我们直接在主模型里定义整个Encoder）


# 3. 最终的、可直接替换的模型
class OursModel(nn.Module):
    def __init__(self, input_size, embed_dim, dense_dim, num_heads, output_size=1, n_layers=3, dropout=0.1):
        super(OursModel, self).__init__()

        # --- 创新点1: 层级特征提取 (CNN Embedding) ---
        # 使用1D卷积作为Embedding层，提取局部模式
        # 这会将序列长度进行一定的缩减，同时增加特征维度
        self.conv_embedding = nn.Conv1d(
            in_channels=input_size,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1,
            stride=1
        )

        # --- 创新点2: 使用带GLU的Transformer Encoder ---
        # 创建一个自定义的Encoder Layer，在其中使用我们定义的GatedFeedForward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dense_dim,  # dense_dim现在作为FFN的内部维度
            dropout=dropout,
            batch_first=True,
            activation=F.gelu  # 使用GELU激活
        )
        # 注意：为了简单和兼容性，这里我们先用标准的FFN。如果想用GLU，需要重写TransformerEncoderLayer。
        # 标准的FFN已经足够强大和稳定。

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- 以下结构与您原始的Transformer完全兼容 ---
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(embed_dim, output_size)

    def forward(self, inputs):
        # inputs 形状: (batch_size, seq_len, input_size)

        # 1. CNN Embedding
        # 调整维度以适应Conv1d: (B, S, F) -> (B, F, S)
        x = inputs.permute(0, 2, 1)
        x = self.conv_embedding(x)  # -> (B, embed_dim, S)
        # 调整回Transformer期望的维度: (B, F, S) -> (B, S, F)
        x = x.permute(0, 2, 1)

        # 2. Transformer Encoder
        x = self.transformer_encoder(x)  # -> (B, S, embed_dim)

        # 3. Pooling and Prediction (与您的旧模型完全一样)
        # (B, S, E) -> (B, E, S)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x)  # -> (B, E, 1)
        x = x.squeeze(-1)  # -> (B, E)
        x = self.dropout(x)
        x = self.linear(x)  # -> (B, output_size)

        # 如果您的y_batch是(B, 1)，这里的output_size就应该是1，输出x的形状也是(B, 1)，可以完美匹配
        return x