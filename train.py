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
from models.LSTM import LSTMModel  # 导入 LSTM 模型
from models.Transformer import TransformerModel  # 导入 Transformer 模型
from config import get_parser  # 导入 Config 类
import datetime
import untils
from torch.utils.tensorboard import SummaryWriter

def sliding_window(dataset, time_steps=1, predict_future=False):
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
    X_train, y_train = sliding_window(train_scaled, TIME_STEPS)
    # 创建测试集时，不进行滑动窗口，保留所有数据用于预测
    X_test, y_test = sliding_window(test_scaled, TIME_STEPS, predict_future=True)

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
    logger = untils.create_logger(output_dir)

    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'runs'))

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
        global_step = 0
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

                # 写入 TensorBoard
                writer.add_scalar('Loss/train', loss.item(), global_step)
                global_step += 1

        writer.close()

        # 8. 保存模型
        weight_dir = './weight'
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
        torch.save(model.state_dict(), os.path.join(weight_dir, f'{args.model.lower()}_model.pth'))

    # 4.3 模型预测
    # model.eval()
    # with torch.no_grad():
    #     predictions = []
    #     input_seq = torch.tensor(test_scaled[:TIME_STEPS], dtype=torch.float32).unsqueeze(0)
    #     num_features = input_seq.shape[2]
    #     TIME_STEPS = input_seq.shape[1]
    #     total_steps = len(test_scaled)
    #     steps = 0
    #     while len(predictions) < total_steps:
    #         output = model(input_seq)
    #         predictions.append(output.item())
    #         if len(predictions) == total_steps:
    #             break
    #         # 构造新输入：[1, 1, num_features]
    #         if steps + TIME_STEPS < total_steps:
    #             # 用真实特征
    #             next_input = torch.tensor(test_scaled[steps + TIME_STEPS], dtype=torch.float32).reshape(1, 1, num_features)
    #         else:
    #             # 用最后一个真实特征填充
    #             next_input = torch.tensor(test_scaled[-1], dtype=torch.float32).reshape(1, 1, num_features)
    #         # 用预测值替换第一个特征
    #         next_input[0, 0, 0] = output.item()
    #         input_seq = torch.cat((input_seq, next_input), dim=1)[:, -TIME_STEPS:, :]
    #         steps += 1
    #     predictions = np.array(predictions)

    # 4.3 模型预测 —— 直接用滑动窗口批量预测
    model.eval()
    with torch.no_grad():
        # X_test 已经是 (N, TIME_STEPS, num_features)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)  # 👉 这里用 sliding_window 得到的 X_test
        preds_scaled = model(X_test_tensor).squeeze(-1).cpu().numpy()  # (N,)

    # 5.1 逆缩放
    # 因为 scaler.fit_transform 是对所有 feature 做的，这里我们只预测第 0 列，其它列填 0
    dummy = np.zeros((len(preds_scaled), X_train.shape[2] - 1))
    pred_full = np.concatenate([preds_scaled.reshape(-1, 1), dummy], axis=1)
    predictions = scaler.inverse_transform(pred_full)[:, 0]

    # 真值也要对齐：滑窗后 y_test 对应的是 test_data 从 TIME_STEPS 开始的部分
    y_test_original = test_data['Global_active_power'].values[TIME_STEPS:]

    # # 5. 结果评估
    # # 5.1 逆缩放
    # dummy_array = np.zeros((len(predictions), X_train.shape[2] - 1))
    # predictions_full = np.concatenate((predictions.reshape(-1,1), dummy_array), axis=1)
    # predictions = scaler.inverse_transform(predictions_full)[:, 0]
    #
    # # 获取原始测试集的 Global_active_power
    # y_test_original = test_data['Global_active_power'].values
    #
    # # 5.2 计算 MSE 和 MAE
    # MSE = mean_squared_error(y_test_original, predictions)
    # MAE = mean_absolute_error(y_test_original, predictions)
    # print(f'{args.model} (Test {args.predict_days}) MSE: {MSE}, MAE: {MAE}')
    # logger.info(f'{args.model} (Test {args.predict_days}) MSE: {MSE}, MAE: {MAE}')
    #
    # # 6. 结果可视化
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test_original, label='Actual')
    # plt.plot(predictions, label=f'{args.model} Predicted')
    # plt.title(f'{args.model} (Test {args.predict_days}): Actual vs Predicted Global Active Power')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Global Active Power')
    # plt.legend()
    #
    # # 保存预测值和真实值
    # lstm_results = np.column_stack((y_test_original, predictions))
    # output_file_name = f'{args.model.lower()}_predictions_{args.predict_days}.csv'
    # np.savetxt(os.path.join(output_dir, output_file_name), lstm_results, delimiter=',', header='Actual,Predicted', comments='')
    # plt.savefig(os.path.join(output_dir, f'{args.model.lower()}_predictions_{args.predict_days}.png'))
    # plt.close()

    # -------------------------------
    # 5.1 逆缩放 —— 得到 predictions（预测值）
    # -------------------------------
    dummy = np.zeros((len(preds_scaled), X_train.shape[2] - 1))
    pred_full = np.concatenate([preds_scaled.reshape(-1, 1), dummy], axis=1)
    predictions = scaler.inverse_transform(pred_full)[:, 0]  # shape=(N,)

    # -------------------------------
    # 5.2 真值对齐 —— 得到 y_test_original
    # -------------------------------
    # test_data 是原始未归一化的 DataFrame
    # 我们前面用 sliding_window 删掉了前 TIME_STEPS 个值
    y_test_original = test_data['Global_active_power'].values[TIME_STEPS:]  # shape=(N,)

    # -------------------------------
    # 5.3 评估指标 —— 确保用到了 y_test_original
    # -------------------------------
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    MSE = mean_squared_error(y_test_original, predictions)
    MAE = mean_absolute_error(y_test_original, predictions)
    print(f'{args.model} (Test {args.predict_days}) → MSE: {MSE:.4f}, MAE: {MAE:.4f}')
    logger.info(f'{args.model} (Test {args.predict_days}) → MSE: {MSE:.4f}, MAE: {MAE:.4f}')

    # -------------------------------
    # 5.4 保存到 CSV —— 同时保存真值和预测值
    # -------------------------------

    df_res = pd.DataFrame({
        'Actual': y_test_original,
        'Predicted': predictions
    })
    output_file = os.path.join(output_dir, f'{args.model.lower()}_results_{args.predict_days}.csv')
    df_res.to_csv(output_file, index=False)

    # -------------------------------
    # 6. 结果可视化 —— 用 y_test_original 画出真值曲线
    # -------------------------------
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original, label='Actual')  # 真值
    plt.plot(predictions, label=f'{args.model} Predicted')  # 预测
    plt.title(f'{args.model} (Test {args.predict_days}): Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Global Active Power')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{args.model.lower()}_plot_{args.predict_days}.png'))
    plt.show()


if __name__ == "__main__":
    args = get_parser()
    main(args)
