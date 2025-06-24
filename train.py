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
from config import Config  # 导入 Config 类

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

    test_size_90 = args.test_size_90  # 测试集1：前90天
    test_data_90 = test_data[:test_size_90]

    test_size_365 = args.test_size_365  # 测试集2：前365天
    test_data_365 = test_data[:test_size_365]

    train_scaled = scaler.fit_transform(train_data[features])
    test_scaled_90 = scaler.transform(test_data_90[features])
    test_scaled_365 = scaler.transform(test_data_365[features])

    TIME_STEPS = args.time_steps
    X_train, y_train = create_dataset(train_scaled, TIME_STEPS)
    # 创建测试集时，不进行滑动窗口，保留所有数据用于预测
    X_test_90, y_test_90 = create_dataset(test_scaled_90, TIME_STEPS, predict_future=True)
    X_test_365, y_test_365 = create_dataset(test_scaled_365, TIME_STEPS, predict_future=True)

    # 1.6 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test_90 = torch.tensor(X_test_90, dtype=torch.float32)
    y_test_90 = torch.tensor(y_test_90, dtype=torch.float32).reshape(-1, 1)
    X_test_365 = torch.tensor(X_test_365, dtype=torch.float32)
    y_test_365 = torch.tensor(y_test_365, dtype=torch.float32).reshape(-1, 1)

    # 1.7 创建数据加载器
    batch_size = args.batch_size
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. LSTM 模型构建
    input_size = X_train.shape[2]
    hidden_size = args.hidden_size
    output_size = args.output_size
    model_lstm = LSTMModel(input_size, hidden_size, output_size)

    # 3. Transformer 模型构建
    # embed_dim = args.embed_dim
    # dense_dim = args.dense_dim
    # num_heads = args.num_heads
    # model_transformer = TransformerModel(input_size, embed_dim, dense_dim, num_heads, output_size)

    # 4. 模型训练与预测
    # 4.0 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=args.learning_rate)
    # optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=args.learning_rate)

    # 加载模型
    if args.load_model:
        model_lstm.load_state_dict(torch.load(args.model_path))
        print("已加载模型！")
    else:
        # 4.1 LSTM 训练
        num_epochs = args.num_epochs
        model_lstm.train()
        for epoch in range(num_epochs):
            for X_batch, y_batch in train_loader:
                optimizer_lstm.zero_grad()
                outputs = model_lstm(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer_lstm.step()

        # 4.2 Transformer 训练
        # model_transformer.train()
        # for epoch in range(num_epochs):
        #     for X_batch, y_batch in train_loader:
        #         optimizer_transformer.zero_grad()
        #         outputs = model_transformer(X_batch)
        #         loss = criterion(outputs, y_batch)
        #         loss.backward()
        #         optimizer_transformer.step()

    # 4.3 LSTM 预测
    model_lstm.eval()
    with torch.no_grad():
        # 使用整个测试集进行预测
        X_test_90_tensor = torch.tensor(test_scaled_90[TIME_STEPS:], dtype=torch.float32).unsqueeze(1)
        X_test_365_tensor = torch.tensor(test_scaled_365[TIME_STEPS:], dtype=torch.float32).unsqueeze(1)
        lstm_predictions_90 = model_lstm(X_test_90_tensor).numpy()
        lstm_predictions_365 = model_lstm(X_test_365_tensor).numpy()
        
        # final_predictions_90 = predict_full_sequence(model_lstm, test_scaled_90, TIME_STEPS)
        # final_predictions_365 = predict_full_sequence(model_lstm, test_scaled_365, TIME_STEPS)
        # # 合并预测结果
        # lstm_predictions_90 = np.concatenate((lstm_predictions_90, final_predictions_90))
        # lstm_predictions_365 = np.concatenate((lstm_predictions_365, final_predictions_365))

    # 4.4 Transformer 预测
    # model_transformer.eval()
    # with torch.no_grad():
    #     transformer_predictions_90 = model_transformer(X_test_90).numpy()
    #     transformer_predictions_365 = model_transformer(X_test_365).numpy()

    # 5. 结果评估
    # 5.1 逆缩放
    # 创建与测试集大小相同的全零数组
    dummy_array_90 = np.zeros((len(test_scaled_90) - TIME_STEPS, X_train.shape[2] - 1))
    dummy_array_365 = np.zeros((len(test_scaled_365) - TIME_STEPS, X_train.shape[2] - 1))

    # 将预测结果和全零数组拼接
    lstm_predictions_90_full = np.concatenate((lstm_predictions_90, dummy_array_90), axis=1)
    lstm_predictions_365_full = np.concatenate((lstm_predictions_365, dummy_array_365), axis=1)

    # 逆缩放
    lstm_predictions_90 = scaler.inverse_transform(lstm_predictions_90_full)[:, 0]
    lstm_predictions_365 = scaler.inverse_transform(lstm_predictions_365_full)[:, 0]

    # 获取原始测试集的 Global_active_power
    y_test_original_90 = test_data_90['Global_active_power'][TIME_STEPS:].values
    y_test_original_365 = test_data_365['Global_active_power'][TIME_STEPS:].values

    # 5.2 计算 MSE 和 MAE (Test set 1: 90 days)
    lstm_mse_90 = mean_squared_error(y_test_original_90, lstm_predictions_90)
    lstm_mae_90 = mean_absolute_error(y_test_original_90, lstm_predictions_90)
    # transformer_mse_90 = mean_squared_error(y_test_original_90, transformer_predictions_90)
    # transformer_mae_90 = mean_absolute_error(y_test_original_90, transformer_predictions_90)

    print(f'LSTM (Test 90) MSE: {lstm_mse_90}, MAE: {lstm_mae_90}')
    # print(f'Transformer (Test 90) MSE: {transformer_mse_90}, MAE: {transformer_mae_90}')

    # 5.3 计算 MSE 和 MAE (Test set 2: 365 days)
    lstm_mse_365 = mean_squared_error(y_test_original_365, lstm_predictions_365)
    lstm_mae_365 = mean_absolute_error(y_test_original_365, lstm_predictions_365)
    # transformer_mse_365 = mean_squared_error(y_test_original_365, transformer_predictions_365)
    # transformer_mae_365 = mean_absolute_error(y_test_original_365, transformer_predictions_365)

    print(f'LSTM (Test 365) MSE: {lstm_mse_365}, MAE: {lstm_mae_365}')
    # print(f'Transformer (Test 365) MSE: {transformer_mse_365}, MAE: {transformer_mae_365}')

    # 6. 结果可视化
    # 6.1 LSTM (Test set 1: 90 days)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original_90, label='Actual')
    plt.plot(lstm_predictions_90, label='LSTM Predicted')
    plt.title('LSTM (Test 90): Actual vs Predicted Global Active Power')
    plt.xlabel('Time Steps')
    plt.ylabel('Global Active Power')
    plt.legend()

    # 7. 结果保存
    lstm_output_dir = './output/lstm_results'
    if not os.path.exists(lstm_output_dir):
        os.makedirs(lstm_output_dir)

    # 保存 LSTM 预测值和真实值 (Test set 1: 90 days)
    lstm_results_90 = np.column_stack((y_test_original_90, lstm_predictions_90))
    np.savetxt(os.path.join(lstm_output_dir, 'lstm_predictions_90.csv'), lstm_results_90, delimiter=',', header='Actual,Predicted', comments='')
    plt.savefig(os.path.join(lstm_output_dir, 'lstm_predictions_90.png'))
    plt.close()

    # 6.2 Transformer (Test set 1: 90 days)
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test_original_90, label='Actual')
    # plt.plot(transformer_predictions_90, label='Transformer Predicted')
    # plt.title('Transformer (Test 90): Actual vs Predicted Global Active Power')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Global Active Power')
    # plt.legend()
    # plt.savefig(os.path.join(lstm_output_dir, 'transformer_predictions_90.png'))
    # plt.close()

    # 6.3 LSTM (Test set 2: 365 days)
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original_365, label='Actual')
    plt.plot(lstm_predictions_365, label='LSTM Predicted')
    plt.title('LSTM (Test 365): Actual vs Predicted Global Active Power')
    plt.xlabel('Time Steps')
    plt.ylabel('Global Active Power')
    plt.legend()

    # 保存 LSTM 预测值和真实值 (Test set 2: 365 days)
    lstm_results_365 = np.column_stack((y_test_original_365, lstm_predictions_365))
    np.savetxt(os.path.join(lstm_output_dir, 'lstm_predictions_365.csv'), lstm_results_365, delimiter=',', header='Actual,Predicted', comments='')
    plt.savefig(os.path.join(lstm_output_dir, 'lstm_predictions_365.png'))
    plt.close()

    # 6.4 Transformer (Test set 2: 365 days)
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test_original_365, label='Actual')
    # plt.plot(transformer_predictions_365, label='Transformer Predicted')
    # plt.title('Transformer (Test 365): Actual vs Predicted Global Active Power')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Global Active Power')
    # plt.legend()
    # plt.savefig(os.path.join(lstm_output_dir, 'transformer_predictions_365.png'))
    # plt.close()

    # 保存 LSTM 和 Transformer 的评估指标
    with open(os.path.join(lstm_output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f'LSTM (Test 90) MSE: {lstm_mse_90}, MAE: {lstm_mae_90}\n')
        # f.write(f'Transformer (Test 90) MSE: {transformer_mse_90}, MAE: {transformer_mae_90}\n')
        f.write(f'LSTM (Test 365) MSE: {lstm_mse_365}, MAE: {lstm_mae_365}\n')
        # f.write(f'Transformer (Test 365) MSE: {transformer_mse_365}, MAE: {transformer_mae_365}\n')

    # 7. 保存数据
    np.savetxt(os.path.join(lstm_output_dir, 'lstm_results_90.txt'), lstm_predictions_90, header=f'LSTM MSE: {lstm_mse_90}, MAE: {lstm_mae_90}')
    # np.savetxt(os.path.join(lstm_output_dir, 'transformer_results_90.txt'), transformer_predictions_90, header=f'Transformer MSE: {transformer_mse_90}, MAE: {transformer_mae_90}')
    np.savetxt(os.path.join(lstm_output_dir, 'lstm_results_365.txt'), lstm_predictions_365, header=f'LSTM MSE: {lstm_mse_365}, MAE: {lstm_mae_365}')
    # np.savetxt(os.path.join(lstm_output_dir, 'transformer_results_365.txt'), transformer_predictions_365, header=f'Transformer MSE: {transformer_mse_365}, MAE: {transformer_mae_365}')

    # 8. 保存模型
    weight_dir = './weight'
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    torch.save(model_lstm.state_dict(), os.path.join(weight_dir, 'lstm_model.pth'))
    # torch.save(model_transformer.state_dict(), os.path.join(weight_dir, 'transformer_model.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM and Transformer models for power prediction.")
    config = Config()
    parser.add_argument('--time_steps', type=int, default=config.time_steps, help='Number of time steps for the sliding window.')
    parser.add_argument('--hidden_size', type=int, default=config.hidden_size, help='Hidden size for the LSTM model.')
    parser.add_argument('--embed_dim', type=int, default=config.embed_dim, help='Embedding dimension for the Transformer model.')
    parser.add_argument('--dense_dim', type=int, default=config.dense_dim, help='Dense dimension for the Transformer model.')
    parser.add_argument('--num_heads', type=int, default=config.num_heads, help='Number of heads for the Transformer model.')
    parser.add_argument('--output_size', type=int, default=config.output_size, help='Output size for the models.')
    parser.add_argument('--num_epochs', type=int, default=config.num_epochs, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size for training.')
    parser.add_argument('--train_size', type=int, default=config.train_size, help='Size of the training dataset (last N days).')
    parser.add_argument('--test_size_90', type=int, default=config.test_size_90, help='Size of the first test dataset (first N days).')
    parser.add_argument('--test_size_365', type=int, default=config.test_size_365, help='Size of the second test dataset (first N days).')
    parser.add_argument('--learning_rate', type=float, default=config.learning_rate, help='Learning rate for the optimizers.')
    parser.add_argument('--dropout_rate', type=float, default=config.dropout_rate, help='Dropout rate for the Transformer model.')
    parser.add_argument('--load_model', action='store_true', help='是否加载模型')
    parser.add_argument('--model_path', type=str, default='./weight/lstm_model.pth', help='模型路径')

    args = parser.parse_args()
    main(args)
