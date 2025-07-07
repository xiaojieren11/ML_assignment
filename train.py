import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from models.LSTM import LSTMModel  # 导入 LSTM 模型
from models.Transformer import TransformerModel 
from models.ours import OursModel
from models.ours1 import CNNTransformer  # 导入 Ours1 模型
from config import get_parser  
import datetime
import untils
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

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

def create_model(model_type, input_size, hidden_size, output_size, embed_dim=None, dense_dim=None, num_heads=None):
    if args.model == 'LSTM':
        model = LSTMModel(input_size, hidden_size, output_size)
    elif args.model == 'Transformer':
        embed_dim = args.embed_dim
        dense_dim = args.dense_dim
        num_heads = args.num_heads
        model = TransformerModel(input_size, embed_dim, dense_dim, num_heads, output_size)
    elif args.model == 'Ours':
        model = OursModel(
            input_size=input_size,  # 您数据中的总特征数 (e.g., 13)
            embed_dim=128,  # 示例值
            dense_dim=256,  # 示例值
            num_heads=8,  # 示例值
            output_size=1,  # 因为您的y_batch是(B,1)，所以这里必须是1
            n_layers=3,  # 示例值
            dropout=0.1
        )
    elif args.model == 'Ours1':
        model = CNNTransformer(
            input_dim=input_size,  # 输入特征数
            model_dim=128,  # 模型维度
            num_heads=8,  # 注意力头数
            num_layers=3,  # Transformer 层数
            output_dim=output_size,  # 输出维度  
        )
    else:
        raise ValueError("Invalid model type. Choose 'LSTM' or 'Transformer'.")
    
    return model

def model_train(model, train_loader, val_loader, criterion, optimizer, num_epochs, num_steps, logger, writer, device):
    print(next(model.parameters()).device)
    model.train()
    start_time = datetime.datetime.now()
    global_step = 0
    best_val_loss = float('inf')
    best_model_path = None

    for epoch in range(num_epochs):
        model.train()
        for idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
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

        # 每隔5个epoch进行一次验证集评估
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    val_losses.append(val_loss.item())
            avg_val_loss = np.mean(val_losses)
            print(f'Validation Loss after epoch {epoch+1}: {avg_val_loss:.4f}')
            logger.info(f'Validation Loss after epoch {epoch+1}: {avg_val_loss:.4f}')
            writer.add_scalar('Loss/val', avg_val_loss, epoch)

            # 保存效果最好的模型
            weight_dir = './weight'
            if not os.path.exists(weight_dir):
                os.makedirs(weight_dir)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(weight_dir, f'{args.model.lower()}_best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at epoch {epoch+1} with val_loss {best_val_loss:.4f}")

    writer.close()
    # 训练结束后加载最佳模型
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))

def main(args):
    # 0. 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # 1. 数据准备
    # 1.1 读取数据
    train_data = pd.read_csv('./dataset/train_processed.csv', index_col='DateTime', parse_dates=True)
    test_data = pd.read_csv('./dataset/test_processed.csv', index_col='DateTime', parse_dates=True)

    # 1.2 定义特征列
    features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3','Sub_metering_remainder', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']
    # 1.3 数据缩放
    scaler = MinMaxScaler()

    # 1.4 使用全部训练数据
    # 直接使用全部train_data

    TIME_STEPS = args.time_steps
    #TODO 解决 test_data 的长度问题
    test_size = args.predict_days + TIME_STEPS
    test_data = test_data[:test_size]

    train_scaled = scaler.fit_transform(train_data[features])
    test_scaled = scaler.transform(test_data[features])

    # 1.5 划分训练集和验证集
    X_train, y_train = sliding_window(train_scaled, TIME_STEPS)
    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    # 划分训练集和验证集（如8:2）
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    # 创建测试集时，不进行滑动窗口，保留所有数据用于预测
    X_test, y_test = sliding_window(test_scaled, TIME_STEPS, predict_future=True)
    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

    # 1.6 转换为 PyTorch 张量
    X_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_tr, dtype=torch.float32).reshape(-1, 1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)

    # 1.7 创建数据加载器
    batch_size = args.batch_size
    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. 模型构建
    input_size = X_train.shape[2]
    hidden_size = args.hidden_size
    output_size = args.output_size

    model = create_model(
        model_type=args.model,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        embed_dim=args.embed_dim,
        dense_dim=args.dense_dim,
        num_heads=args.num_heads
    ).to(device)

    # 7. 结果保存
    output_dir = os.path.join('./output', f'{args.model.lower()}1_results')
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
        model_train(model, train_loader, val_loader, criterion, optimizer, num_epochs, num_steps, logger, writer, device)

    # 4.3 模型预测 —— 直接用滑动窗口批量预测
    model.eval()
    # 若训练阶段保存了最佳模型，则此处已加载
    with torch.no_grad():
        # X_test 已经在device上
        preds_scaled = model(X_test).squeeze(-1).cpu().numpy()  # (N,)

    # 5.1 逆缩放
    # 因为 scaler.fit_transform 是对所有 feature 做的，这里我们只预测第 0 列，其它列填 0
    dummy = np.zeros((len(preds_scaled), X_train.shape[2] - 1))
    pred_full = np.concatenate([preds_scaled.reshape(-1, 1), dummy], axis=1)
    predictions = scaler.inverse_transform(pred_full)[:, 0]

    # 真值也要对齐：滑窗后 y_test 对应的是 test_data 从 TIME_STEPS 开始的部分
    y_test_original = test_data['Global_active_power'].values[TIME_STEPS:]

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
    untils.plot(y_test_original, predictions, args, output_dir)


if __name__ == "__main__":
    args = get_parser()
    main(args)
