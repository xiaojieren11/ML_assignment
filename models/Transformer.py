import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super(TransformerEncoder, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, dense_dim),
            nn.ReLU(),
            nn.Linear(dense_dim, embed_dim)
        )
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)

    def forward(self, inputs):
        attention_output, _ = self.attention(inputs, inputs, inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

class TransformerModel(nn.Module):
    def __init__(self, input_size, embed_dim, dense_dim, num_heads, output_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, embed_dim)  # 添加线性层进行维度转换
        self.transformer_encoder = TransformerEncoder(embed_dim, dense_dim, num_heads)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(embed_dim, output_size)  # 修改线性层的输入维度

    def forward(self, inputs):
        x = self.embedding(inputs)  # 通过线性层进行维度转换
        x = self.transformer_encoder(x)
        x = self.global_avg_pool(x.transpose(1, 2)).squeeze(-1)
        x = self.dropout(x)
        x = self.linear(x)
        return x