import torch
import torch.nn as nn
import math

# 修改1：添加位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # 修改2：优化前馈网络结构，增加中间维度
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim * 2),
            nn.GELU(),
            nn.Linear(dense_dim * 2, embed_dim),
            nn.Dropout(0.1)
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        attention_output, _ = self.attention(inputs, inputs, inputs)
        # 修改3：调整残差连接顺序
        proj_input = self.layernorm_1(inputs + self.dropout(attention_output))
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + self.dropout(proj_output))

class TransformerModel(nn.Module):
    def __init__(self, input_size, embed_dim, dense_dim, num_heads, output_size):
        super(TransformerModel, self).__init__()
        # 修改4：增加编码层堆叠数量
        self.embedding = nn.Linear(input_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        # 堆叠4个Transformer编码层
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, dense_dim, num_heads)
            for _ in range(4)
        ])
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        # 修改5：增强分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, output_size)
        )

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.pos_encoding(x)
        for encoder in self.transformer_encoders:
            x = encoder(x)
        x = self.global_avg_pool(x.transpose(1, 2)).squeeze(-1)
        x = self.dropout(x)
        return self.classifier(x)