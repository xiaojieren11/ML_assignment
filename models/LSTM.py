import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        # 单层 LSTM，batch_first=True 表示输入维度为 (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=10,
            batch_first=True
        )
        # 从隐藏状态到输出的全连接层
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x 的形状: [batch, seq_len, input_size]
        out_seq, (h_n, c_n) = self.lstm(x)
        # h_n 的形状: [num_layers * num_directions, batch, hidden_size]
        # 因为 num_layers=1 且 单向，所以 h_n[0] 就是最后一个时间步的隐藏状态
        h_last = h_n[0]            # -> [batch, hidden_size]
        y = self.linear(h_last)    # -> [batch, output_size]
        return y