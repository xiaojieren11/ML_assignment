import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import train_test_split  # 新增导入
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from config import get_parser
import utils
import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from models.LSTM import LSTMModel 
from models.Transformer import TransformerModel 
from models.Ours import CNNTransformer 
def load_data(TRAIN_DAYS, PREDICT_DAYS, WIDTH, TARGET_COLUMN, BATCH_SIZE, device):
    # 数据加载
    train_df = pd.read_csv('./dataset/train_processed.csv', index_col='datetime', parse_dates=True)
    test_df = pd.read_csv('./dataset/test_processed.csv', index_col='datetime', parse_dates=True)
    train_df = train_df.tail(TRAIN_DAYS)  
    test_df = test_df.head(PREDICT_DAYS)
    print(f"Train data shape: {train_df.shape}, Test data shape: {test_df.shape}")

    common_cols = list(train_df.columns.intersection(test_df.columns))
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]

    target_column_index = common_cols.index(TARGET_COLUMN)

    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_df)
    scaled_test_data = scaler.transform(test_df)

    target_scaler = MinMaxScaler()
    target_scaler.fit(train_df[[TARGET_COLUMN]])

    X_train, y_train = utils.sliding_window(scaled_train_data, WIDTH, target_column_index)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    combined_for_test_sequences = np.concatenate((scaled_train_data[-WIDTH:], scaled_test_data))
    X_test, y_test = utils.sliding_window(combined_for_test_sequences, WIDTH, target_column_index)
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 使用train_test_split划分训练集和验证集
    train_idx, val_idx = train_test_split(
        list(range(len(train_dataset))),
        test_size=0.2,  # 按照8:2比例划分
        random_state=42
    )
    
    # 创建采样器
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # 修改训练集DataLoader并新增验证集DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"数据集已准备完毕，训练集大小为 {len(train_loader)}，验证集大小为 {len(val_loader)}，测试集大小为 {len(test_loader)}")

    return X_train, train_loader, val_loader, test_loader, target_scaler

def create_model(model_type, input_size, hidden_size, output_size, embed_dim, dense_dim, num_heads):
    if model_type == 'LSTM':
        model = LSTMModel(
            input_size, 
            hidden_size, 
            output_size
        )
    elif model_type == 'Transformer':
        model = TransformerModel(
            input_size,
            embed_dim, 
            dense_dim, 
            num_heads, 
            output_size
        )
    elif model_type == 'Ours':
        model = CNNTransformer(
            input_dim=input_size,  
            model_dim=64,  
            num_heads=4,  
            num_layers=2,  
            output_dim=output_size,  
        )
    else:
        raise ValueError("Invalid model type. Choose 'LSTM', 'Transformer' or 'Ours'.")
    return model

def model_train(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, args, logger, writer):
    # 训练配置
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    start_time = datetime.datetime.now()
    num_steps = len(train_loader)

    weight_dir = './weight'
    best_val_loss = float('inf')
    best_model_path = os.path.join(weight_dir, f'{args.model.lower()}_best_model_{args.predict_days}.pth')

    for epoch in range(EPOCHS):
        # 训练阶段 - 新增loss累加
        total_loss = 0.0
        for idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            now = datetime.datetime.now()
            time_diff = now - start_time
            etas = time_diff.total_seconds() / (epoch * num_steps + idx + 1) * (EPOCHS * num_steps - epoch * num_steps - idx - 1)
            print(
                f'Train: [{epoch+1}/{EPOCHS}][{idx+1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'loss {loss.item():.4f}\t'
            )
            logger.info(
                f'Train: [{epoch+1}/{EPOCHS}][{idx+1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'loss {loss.item():.4f}\t'
            )
            # 累加loss
            total_loss += loss.item() * X_batch.size(0)

        # 每个epoch记录训练损失
        epoch_loss = total_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # 验证阶段保持不变
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                outputs = model(X_val)
                val_loss += criterion(outputs, y_val).item() * X_val.size(0)
        val_loss /= len(val_loader.dataset)
        writer.add_scalar('Loss/val', val_loss, epoch)
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        
        model.train()

    writer.close()
    model.load_state_dict(torch.load(best_model_path))

def model_eval(model, test_loader, target_scaler):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    predictions_rescaled = target_scaler.inverse_transform(predictions)
    actuals_rescaled = target_scaler.inverse_transform(actuals)
    mse = np.mean((predictions_rescaled - actuals_rescaled) ** 2)
    mae = np.mean(np.abs(predictions_rescaled - actuals_rescaled))
    return predictions_rescaled, actuals_rescaled, mse, mae

def main(args):
    # 超参数提取
    WIDTH = args.win_width
    TARGET_COLUMN = 'Global_active_power'
    NUM_RUNS = 5
    EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    PREDICT_DAYS = args.predict_days
    TRAIN_DAYS = args.train_days

    # 设备选取
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 结果保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(f'./output/{args.model.lower()}_results_{args.predict_days}', f'{timestamp}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    logger = utils.create_logger(output_dir)
    log_dir=os.path.join(output_dir, 'runs')
    writer = SummaryWriter(log_dir)

    X_train, train_loader, val_loader, test_loader, target_scaler = load_data(TRAIN_DAYS, PREDICT_DAYS, WIDTH, TARGET_COLUMN, BATCH_SIZE, device)

    model = create_model(
            model_type=args.model,
            input_size=X_train.shape[2],
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            embed_dim=args.embed_dim,
            dense_dim=args.dense_dim,
            num_heads=args.num_heads,
    ).to(device)

    # 如果仅测试模型则跳过训练过程
    if args.eval_only:
        model_path = args.model_path
        model.load_state_dict(torch.load(model_path,weights_only=True))
        print("已加载模型！")
    else:
        # 传递验证集DataLoader
        model_train(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, args, logger, writer)
        # 绘制损失图
        utils.plot_loss(log_dir)

    # 模型测试
    predictions_rescaled, actuals_rescaled, mse, mae = model_eval(model, test_loader, target_scaler)
    print("Actuals shape:", actuals_rescaled.shape)
    print("Predictions shape:", predictions_rescaled.shape)
    # 打印结果
    print(f'{args.model} (Test {args.predict_days}) → MSE: {mse:.4f}, MAE: {mae:.4f}')
    logger.info(f'{args.model} (Test {args.predict_days}) → MSE: {mse:.4f}, MAE: {mae:.4f}')

    # 保存结果
    utils.save_res(actuals_rescaled, predictions_rescaled, args, output_dir, mse, mae)
    # 绘制结果
    utils.plot_res(actuals_rescaled, predictions_rescaled, args, output_dir)

if __name__ == '__main__':
    args = get_parser()
    main(args)