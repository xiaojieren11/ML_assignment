import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

def model_train(model, train_loader, EPOCHS, LEARNING_RATE, args, logger, writer):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()

    start_time = datetime.datetime.now()
    global_step = 0
    num_steps = len(train_loader)

    for epoch in range(EPOCHS):
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
            # 写入 TensorBoard
            writer.add_scalar('Loss/train', loss.item(), global_step)
            global_step += 1

    writer.close()
    weight_dir = './weight'
    best_model_path = os.path.join(weight_dir, f'{args.model.lower()}_best_model_{args.predict_days}.pth')
    torch.save(model.state_dict(), best_model_path)

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
    WIDTH = args.win_width
    TARGET_COLUMN = 'Global_active_power'
    NUM_RUNS = 5
    EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    PREDICT_DAYS = args.predict_days

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

    # 数据加载
    train_df = pd.read_csv('./dataset/train_processed.csv', index_col='datetime', parse_dates=True)
    test_df = pd.read_csv('./dataset/test_processed.csv', index_col='datetime', parse_dates=True)

    test_df = test_df.head(PREDICT_DAYS)

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

    combined_for_test_sequences = np.concatenate((scaled_train_data[-WIDTH:], scaled_test_data))
    X_test, y_test = utils.sliding_window(combined_for_test_sequences, WIDTH, target_column_index)

    X_test = X_test[:len(test_df)]
    y_test = y_test[:len(test_df)]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_mse_scores, all_mae_scores = [], []

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
        model_train(model, train_loader, EPOCHS, LEARNING_RATE, args, logger, writer)
        # 绘制损失图
        utils.plot_loss(log_dir)
        # # 训练结束后加载最佳模型
        # if best_model_path:
        #     model.load_state_dict(torch.load(best_model_path))

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