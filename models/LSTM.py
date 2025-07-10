import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, predict_days=90):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=3,
<<<<<<< HEAD
            batch_first=True,
            dropout=0.2
=======
            batch_first=True
>>>>>>> change
        )
        self.linear = nn.Linear(hidden_size, output_size * predict_days)
        self.predict_days = predict_days

    def forward(self, x):
        out_seq, (h_n, c_n) = self.lstm(x)
        y = self.linear(h_n[-1])  # shape: (batch_size, output_size * predict_days)
        # 新增：适配两种输出模式的维度处理
        if self.predict_days == 1:
            return y.unsqueeze(1)  # shape: (batch_size, 1, 1)
        else:
            return y.view(-1, self.predict_days, 1)  # shape: (batch_size, predict_days, 1)
