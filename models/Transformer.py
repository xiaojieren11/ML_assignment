import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, input_size, embed_dim=128, num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, embed_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 100, embed_dim))  
        
        # 简化Transformer层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=6
        )
        
        # 直接取最后一个时间步输出
        self.output_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_enc[:, :x.size(1)]
        x = self.transformer(x)
        return self.output_head(x[:, -1])  # 直接使用最后一步输出