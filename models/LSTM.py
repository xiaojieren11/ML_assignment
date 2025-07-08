import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, predict_days=90):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        # 修改输出层初始化
        self.linear = nn.Linear(hidden_size, output_size * predict_days)
        # 添加正交初始化
        torch.nn.init.orthogonal_(self.linear.weight, gain=0.5)
        self.predict_days = predict_days
        self.output_size = output_size

    def forward(self, x):
        out_seq, (h_n, c_n) = self.lstm(x)
        y = self.linear(h_n[-1])
        return y.view(-1, self.predict_days, self.output_size)
