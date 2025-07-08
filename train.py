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

def sliding_window(dataset, time_steps=1, predict_future=False, predict_days=90):
    X, y = [], []
    if predict_future:
        for i in range(len(dataset) - time_steps - predict_days + 1):
            X.append(dataset[i:(i + time_steps), 1:])
            y.append(dataset[i + time_steps:i + time_steps + predict_days, 0])  # 保持二维形状
    else:
        for i in range(time_steps, len(dataset)):
            X.append(dataset[i-time_steps:i, 1:])
            y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def create_model(model_type, input_size, hidden_size, output_size, embed_dim=None, dense_dim=None, num_heads=None):
    if model_type == 'LSTM':
        model = LSTMModel(input_size, hidden_size, output_size, predict_days=args.predict_days)
    elif model_type == 'Transformer':
        embed_dim = args.embed_dim
        dense_dim = args.dense_dim
        num_heads = args.num_heads
        model = TransformerModel(input_size, embed_dim, dense_dim, num_heads, output_size)
    elif model_type == 'Ours':
        model = OursModel(
            input_size=input_size,  # 您数据中的总特征数 (e.g., 13)
            embed_dim=128,  # 示例值
            dense_dim=256,  # 示例值
            num_heads=8,  # 示例值
            output_size=1,  # 因为您的y_batch是(B,1)，所以这里必须是1
            n_layers=3,  # 示例值
            dropout=0.1
        )
    elif model_type == 'Ours1':
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
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5
    )

    for epoch in range(num_epochs):
        model.train()
        for idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # 添加梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # 检查梯度是否溢出
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > 1e5:
                print(f"Warning: Gradient norm {total_norm} too large at epoch {epoch+1}")
                logger.warning(f"Gradient norm {total_norm} too large at epoch {epoch+1}")
            
            optimizer.step()

            # 记录训练日志
            now = datetime.datetime.now()
            time_diff = now - start_time
            etas = time_diff.total_seconds() / (epoch * num_steps + idx + 1) * (num_epochs * num_steps - epoch * num_steps - idx - 1)
            print(
                f'Train: [{epoch+1}/{num_epochs}][{idx+1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'loss {loss.item():.4f}\t'
                f'grad_norm {total_norm:.2f}'
            )
            logger.info(
                f'Train: [{epoch+1}/{num_epochs}][{idx+1}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))}\t'
                f'loss {loss.item():.4f}\t'
                f'grad_norm {total_norm:.2f}'
            )

            # 写入 TensorBoard
            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('GradNorm/train', total_norm, global_step)
            global_step += 1

        # 每隔5个epoch进行一次验证集评估
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            # 空验证集保护逻辑 - 新增条件判断
            if len(val_loader) == 0:
                print(f"Skip validation: Empty validation set at epoch {epoch+1}")
                logger.info(f"Skip validation: Empty validation set at epoch {epoch+1}")
                continue
                
            model.eval()
            val_losses = []
            with torch.no_grad():
                for X_val, y_val in val_loader:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)
                    val_outputs = model(X_val)
                    
                    # 增强验证输出检查
                    if torch.isnan(val_outputs).any() or torch.isinf(val_outputs).any():
                        raise ValueError(f"模型输出包含NaN/Inf at epoch {epoch+1}")
                    
                    val_loss = criterion(val_outputs, y_val)
                    val_losses.append(val_loss.item())
            
            # 增加验证损失有效性检查
            if not val_losses:
                print(f"Skip validation: Empty validation loss list at epoch {epoch+1}")
                logger.info(f"Skip validation: Empty validation loss list at epoch {epoch+1}")
                continue
                
            avg_val_loss = np.mean(val_losses)
            # 增加inf有效性检查
            if np.isinf(avg_val_loss):
                print(f"Warning: Skip scheduler update due to inf loss at epoch {epoch+1}")
                logger.warning(f"Skip scheduler update due to inf loss at epoch {epoch+1}")
                continue
                
            # 正常更新学习率调度器
            scheduler.step(avg_val_loss)
            
            print(f'Validation Loss after epoch {epoch+1}: {avg_val_loss:.4f}')
            logger.info(f'Validation Loss after epoch {epoch+1}: {avg_val_loss:.4f}')
            writer.add_scalar('Loss/val', avg_val_loss, epoch)

            # 保存效果最好的模型
            weight_dir = './weight'
            if not os.path.exists(weight_dir):
                os.makedirs(weight_dir)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(weight_dir, f'{args.model.lower()}_best_model_{args.predict_days}.pth')
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

    TIME_STEPS = args.time_steps

    # 1.4 训练集划分
    train_size = args.train_size + TIME_STEPS
    train_data = train_data[-train_size:]
    
    test_size = args.predict_days + TIME_STEPS
    test_data = test_data[:test_size]

    train_scaled = scaler.fit_transform(train_data[features])
    test_scaled = scaler.transform(test_data[features])

    # 1.5 划分训练集和验证集
    X_train, y_train = sliding_window(train_scaled, TIME_STEPS, predict_future=True)
    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    # 划分训练集和验证集（如8:2）
    # X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle=True)

    # 创建测试集时，不进行滑动窗口，保留所有数据用于预测
    X_test, y_test = sliding_window(test_scaled, TIME_STEPS, predict_future=True, predict_days=args.predict_days)
    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')  # 应输出(样本数, 90)

    # 添加小样本处理逻辑
    n_samples = X_train.shape[0]
    if n_samples < 2:
        # 小样本场景：直接使用全量数据作为训练集
        X_tr, y_tr = X_train, y_train
        X_val = X_train[-1:]  # 保留最后一个样本作为验证集
        y_val = y_train[-1:]
    else:
        # 动态调整test_size确保至少保留1个验证样本
        test_size = max(1, int(n_samples * 0.2))  # 至少保留1个样本
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, 
            test_size=test_size, 
            random_state=42, 
            shuffle=True
        )
    
    # 1.6 转换为 PyTorch 张量
    X_tr = torch.tensor(X_tr, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    # 修正标签张量形状（自动适配形状）
    y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(-1).to(device)  # (样本数, predict_days, 1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1).to(device) 
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, args.predict_days, 1).to(device)

    # 1.7 创建数据加载器
    batch_size = args.batch_size
    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 新增: 数据有效性验证
    # 检查训练数据是否包含NaN或Inf
    if np.isnan(X_tr.cpu().numpy()).any() or np.isinf(X_tr.cpu().numpy()).any():
        raise ValueError("训练特征数据包含NaN或Inf值")
    if np.isnan(y_tr.cpu().numpy()).any() or np.isinf(y_tr.cpu().numpy()).any():
        raise ValueError("训练标签数据包含NaN或Inf值")
        
    # 检查验证数据是否包含NaN或Inf
    if len(val_loader) > 0:
        X_val_check = next(iter(val_loader))[0].cpu().numpy()
        y_val_check = next(iter(val_loader))[1].cpu().numpy()
        if np.isnan(X_val_check).any() or np.isinf(X_val_check).any():
            raise ValueError("验证特征数据包含NaN或Inf值")
        if np.isnan(y_val_check).any() or np.isinf(y_val_check).any():
            raise ValueError("验证标签数据包含NaN或Inf值")
    
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('./output', f'{args.model.lower()}_results_{args.predict_days}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

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

    # -------------------------------
    # 5.1 逆缩放 —— 得到 predictions（预测值）
    # -------------------------------
    predictions = np.zeros((len(preds_scaled), args.predict_days))

    for j in range(len(preds_scaled)):
        for t in range(args.predict_days):
            # 确定当前预测步对应的实际测试数据索引
            idx = j + TIME_STEPS + t
            if idx >= len(test_scaled):
                idx = len(test_scaled) - 1  # 超出范围时使用最后一个有效索引
            
            # 创建当前样本的特征数组
            current_features = test_scaled[idx].copy()
            current_features[0] = preds_scaled[j, t]  # 替换目标列为预测值
            
            # 进行逆缩放并保存结果
            inversed = scaler.inverse_transform(current_features.reshape(1, -1))
            predictions[j, t] = inversed[0, 0]  # 取出逆缩放后的目标值

    # -------------------------------
    # 5.2 真值对齐 —— 得到 y_test_original
    # -------------------------------
    # 修改为生成连续多步真值
    y_test_original = []
    # 从原始test_data中提取对应位置的真实值
    raw_values = test_data['Global_active_power'].values
    for i in range(len(y_test)):
        start = i
        end = start + args.predict_days
        y_test_original.append(raw_values[start:end])  # 直接使用原始值切片
    y_test_original = np.array(y_test_original)  # (N, 90)
    
    # -------------------------------
    # 5.3 评估指标 —— 确保用到了 y_test_original
    # -------------------------------
    # 将预测结果和真值展平进行评估
    MSE = mean_squared_error(y_test_original.flatten(), predictions.flatten())
    MAE = mean_absolute_error(y_test_original.flatten(), predictions.flatten())
    print(f'{args.model} (Test {args.predict_days}) → MSE: {MSE:.4f}, MAE: {MAE:.4f}')
    logger.info(f'{args.model} (Test {args.predict_days}) → MSE: {MSE:.4f}, MAE: {MAE:.4f}')

    # -------------------------------
    # 5.4 保存到 CSV —— 转换为竖直展示格式
    # -------------------------------
    # 创建竖直格式的DataFrame
    flat_data = []
    for i in range(len(y_test_original)):
        for d in range(args.predict_days):
            flat_data.append({
                'Day': d + 1,
                'Actual': y_test_original[i, d],
                'Predicted': predictions[i, d]
            })
    df_res = pd.DataFrame(flat_data)
    output_file = os.path.join(output_dir, f'{args.model.lower()}_results_{args.predict_days}.csv')
    df_res.to_csv(output_file, index=False)

    # -------------------------------
    # 6. 结果可视化 —— 用 y_test_original 画出真值曲线
    # -------------------------------
    untils.plot(y_test_original, predictions, args, output_dir)


if __name__ == "__main__":
    args = get_parser()
    main(args)
