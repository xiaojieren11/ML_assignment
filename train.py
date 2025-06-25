import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
from models.LSTM import LSTMModel  # 导入 LSTM 模型
from models.Transformer import TransformerModel  # 导入 Transformer 模型
from config import get_parser  # 导入 Config 类
import logging
import datetime

def create_logger(log_dir):
    """创建一个logger，并将日志保存到指定目录下的文件中"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建文件处理器，用于将日志写入文件
    log_file = os.path.join(log_dir, 'Log.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # 创建日志格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到logger
    logger.addHandler(file_handler)

    return logger

def create_dataset(dataset, time_steps=1, predict_future=False):
    X, y = [], []
    if predict_future:
        for i in range(len(dataset) - time_steps):
            X.append(dataset[i:(i + time_steps)])
            y.append(dataset[i + time_steps, 0])
    else:
        for i in range(time_steps, len(dataset)):
            X.append(dataset[i-time_steps:i])
            y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def predict_full_sequence(model, data, time_steps):
    predictions = []
    current_input = data[-time_steps:]   # 使用最后的时间步作为输入
    for _ in range(time_steps):
        prediction = model(torch.tensor(current_input, dtype=torch.float32).unsqueeze(1))
        predictions.append(prediction.item())
        current_input = np.append(current_input[1:], prediction.item())  # 滑动窗口更新
    return predictions


def main(args):
    # 1. 数据准备
    # 1.1 读取数据
    train_data = pd.read_csv('./dataset/train_processed.csv', index_col='DateTime', parse_dates=True)
    test_data = pd.read_csv('./dataset/test_processed.csv', index_col='DateTime', parse_dates=True)

    # 1.2 定义特征列
    features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3','Sub_metering_remainder', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
    # 1.3 数据缩放
    scaler = MinMaxScaler()

    # 1.4 划分训练集和测试集
    train_size = args.train_size  # 训练集取最后90天
    train_data = train_data[-train_size:]

    test_size = args.predict_days
    test_data = test_data[:test_size]

    train_scaled = scaler.fit_transform(train_data[features])
    test_scaled = scaler.transform(test_data[features])

    TIME_STEPS = args.time_steps
    X_train, y_train = create_dataset(train_scaled, TIME_STEPS)
    # 创建测试集时，不进行滑动窗口，保留所有数据用于预测
    X_test, y_test = create_dataset(test_scaled, TIME_STEPS, predict_future=True)

    # 1.6 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # 1.7 创建数据加载器
    batch_size = args.batch_size
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. 模型构建
    input_size = X_train.shape[2]
    hidden_size = args.hidden_size
    output_size = args.output_size

    if args.model == 'LSTM':
        model = LSTMModel(input_size, hidden_size, output_size)
    elif args.model == 'Transformer':
        embed_dim = args.embed_dim
        dense_dim = args.dense_dim
        num_heads = args.num_heads
        model = TransformerModel(input_size, embed_dim, dense_dim, num_heads, output_size)
    else:
        raise ValueError("Invalid model type. Choose 'LSTM' or 'Transformer'.")

    # 7. 结果保存
    output_dir = os.path.join('./output', f'{args.model.lower()}_results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建logger
    logger = create_logger(output_dir)

    # 4. 模型训练与预测
    # 4.0 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 加载模型
    if args.eval_only:
        model.load_state_dict(torch.load(args.model_path,weights_only=True))
        print("已加载模型！")
    else:
        # 4.1 模型训练
        num_epochs = args.num_epochs
        num_steps = len(train_loader)
        model.train()
        start_time = datetime.datetime.now()
        for epoch in range(num_epochs):
            for idx, (X_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                # 记录训练日志
                now = datetime.datetime.now()
                time_diff = now - start_time
                etas = time_diff.total_seconds() / (epoch * num_steps + idx + 1) * (num_epochs * num_steps - epoch * num_steps - idx - 1)
                print(
                    f'Train: [{epoch+1}/{num_epochs}][{idx+1}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))}\t'
                    f'loss {loss.item():.4f}\t'
                )
                logger.info(
                    f'Train: [{epoch+1}/{num_epochs}][{idx+1}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))}\t'
                    f'loss {loss.item():.4f}\t'
                )

        # 8. 保存模型
        weight_dir = './weight'
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        torch.save(model.state_dict(), os.path.join(weight_dir, f'{args.model.lower()}_model.pth'))

    # 4.3 模型预测
    model.eval()
    with torch.no_grad():
        # 使用整个测试集进行预测
        X_test_tensor = torch.tensor(test_scaled[TIME_STEPS:], dtype=torch.float32).unsqueeze(1)
        predictions = model(X_test_tensor).numpy()

    # 5. 结果评估
    # 5.1 逆缩放
    # 创建与测试集大小相同的全零数组
    dummy_array = np.zeros((len(test_scaled) - TIME_STEPS, X_train.shape[2] - 1))

    # 将预测结果和全零数组拼接
    predictions_full = np.concatenate((predictions, dummy_array), axis=1)

    # 逆缩放
    predictions = scaler.inverse_transform(predictions_full)[:, 0]

    # 获取原始测试集的 Global_active_power
    y_test_original = test_data['Global_active_power'][TIME_STEPS:].values

    # 5.2 计算 MSE 和 MAE
    MSE = mean_squared_error(y_test_original, predictions)
    MAE = mean_absolute_error(y_test_original, predictions)
    print(f'{args.model} (Test {args.predict_days}) MSE: {MSE}, MAE: {MAE}')
    logger.info(f'{args.model} (Test {args.predict_days}) MSE: {MSE}, MAE: {MAE}')

    # 6. 结果可视化
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original, label='Actual')
    plt.plot(predictions, label=f'{args.model} Predicted')
    plt.title(f'{args.model} (Test {args.predict_days}): Actual vs Predicted Global Active Power')
    plt.xlabel('Time Steps')
    plt.ylabel('Global Active Power')
    plt.legend()

    # 保存预测值和真实值
    lstm_results = np.column_stack((y_test_original, predictions))
    output_file_name = f'{args.model.lower()}_predictions_{args.predict_days}.csv'
    np.savetxt(os.path.join(output_dir, output_file_name), lstm_results, delimiter=',', header='Actual,Predicted', comments='')
    plt.savefig(os.path.join(output_dir, f'{args.model.lower()}_predictions_{args.predict_days}.png'))
    plt.close()

if __name__ == "__main__":
    args = get_parser()
    main(args)
