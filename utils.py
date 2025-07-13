import logging
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

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

def sliding_window(data, sequence_length, target_column_index):
    sequences, targets = [], []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length, target_column_index])
    return np.array(sequences), np.array(targets)

def plot_res(actuals_rescaled, predictions_rescaled, args, output_dir):
    plt.clf()  # 清除之前的绘图状态
    plt.rcParams.update(plt.rcParamsDefault)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    num_plotted_days = len(actuals_rescaled)
    day_numbers = np.arange(1, num_plotted_days + 1)

    # 图例
    actual_line, = ax.plot(day_numbers, actuals_rescaled, label='Actual Values', color='royalblue', linewidth=2)
    predicted_line, = ax.plot(day_numbers, predictions_rescaled, label='Predicted Values', color='orangered', linestyle='--', linewidth=2)

    ax.set_title(f'Global Active Power: Actual vs. Predicted ({num_plotted_days}-Day Forecast)', fontsize=16)
    ax.set_xlabel(f'Forecast Day Number', fontsize=12)
    ax.set_ylabel('Global Active Power (kW)', fontsize=12)

    # 手动控制图例的显示
    ax.legend(handles=[actual_line, predicted_line], labels=['Actual Values', 'Predicted Values'], fontsize=12)
    
    ax.grid(True)
    plt.tight_layout()

    plot_filename = os.path.join(output_dir, f'{args.model.lower()}_plot_{args.predict_days}.png')
    plt.savefig(plot_filename)

def plot_loss(log_dir):
    # 读取事件文件
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    # 提取训练损失和验证损失
    train_losses = ea.Scalars('Loss/train')
    val_losses = ea.Scalars('Loss/val')  # 新增验证损失读取
    
    # 提取步数和值
    train_steps = [item.step for item in train_losses]
    train_values = [item.value for item in train_losses]
    val_steps = [item.step for item in val_losses]  # 新增验证损失步数
    val_values = [item.value for item in val_losses]  # 新增验证损失值

    # 绘制 loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_values, label='Training Loss')
    plt.plot(val_steps, val_values, label='Validation Loss')  # 新增验证损失曲线
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')  # 更新标题
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'loss_curve.png'))  # 保存图像
    plt.show()

def save_res(actuals_rescaled, predictions_rescaled, args, output_dir, mse, mae):
    flat_data = []
    for i in range(len(predictions_rescaled)):
        flat_data.append({
            'Day': i + 1,
            'Actual':actuals_rescaled[i][0],
            'Predicted': predictions_rescaled[i][0]
        })
    df_res = pd.DataFrame(flat_data)
    output_file = os.path.join(output_dir, f'{args.model.lower()}_results_{args.predict_days}.csv')
    df_res.to_csv(output_file, index=False)

    results_df = pd.DataFrame([{
        'Model': args.model,
        'Predict_Days': args.predict_days,
        'MSE': mse,
        'MAE': mae
    }])
    summary_file = os.path.join(f'./output/{args.model.lower()}_results_{args.predict_days}', 'evaluation_metrics.csv')
    mode = 'a' if os.path.exists(summary_file) else 'w'
    header = mode == 'w'
    results_df.to_csv(summary_file, index=False, mode=mode, header=header)