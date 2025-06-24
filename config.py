# 超参数配置

class Config:
    def __init__(self):
        self.time_steps = 30
        self.hidden_size = 32  # 减小隐藏层大小
        self.embed_dim = 32  # 减小嵌入维度
        self.dense_dim = 16  # 减小 dense_dim
        self.num_heads = 4
        self.output_size = 1
        self.num_epochs = 10
        self.batch_size = 16  # 减小批量大小
        self.train_size = 90
        self.test_size_90 = 90
        self.test_size_365 = 365
        self.learning_rate = 0.0001  # 减小学习率
        self.dropout_rate = 0.5  # 增加 dropout_rate
