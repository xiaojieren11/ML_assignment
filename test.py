from config import get_parser
import utils as utils
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

train_data = pd.read_csv('./dataset/train_processed.csv', index_col='DateTime', parse_dates=True)
test_data = pd.read_csv('./dataset/test_processed.csv', index_col='DateTime', parse_dates=True)

# 1.2 定义特征列
features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3','Sub_metering_remainder', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
# 1.3 数据缩放
scaler = MinMaxScaler()
args = get_parser()

TIME_STEPS = args.time_steps

# 1.4 训练集划分
train_size = args.train_size
train_data = train_data[-train_size:]

test_size = args.predict_days
test_data = test_data[:test_size]

train_scaled = scaler.fit_transform(train_data[features])
test_scaled = scaler.transform(test_data[features])

# 1.5 划分训练集和验证集
X_train, y_train = utils.sliding_window(train_scaled, TIME_STEPS)
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}') 
print("X_train sample:", X_train[:2])
print("y_train sample:", y_train[:2])