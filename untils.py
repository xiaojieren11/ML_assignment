import logging
import os
import matplotlib.pyplot as plt
import numpy as np

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

def sliding_window(dataset, width, step=1):
    """滑动窗口生成训练样本，按指定宽度和步长划分数据"""
    X, y = [], []
    # 修改循环条件：使用 len(dataset)+1 确保包含完整窗口
    for i in range(width, len(dataset)+1, step):
        X.append(dataset[i-width:i, 1:])  # 特征不包含目标列
        y.append(dataset[i-1, 0])  # 单天标签（修正索引对齐）
    return np.array(X), np.array(y)

# 修改绘图函数中的维度验证逻辑
def plot(y_test_original, predictions, args, output_dir):
    plt.figure(figsize=(12, 6))
    
    # 修改展平逻辑，适配新数据结构
    y_flat = y_test_original.reshape(-1)  # 保持原有展平方式
    pred_flat = predictions.reshape(-1)
    
    # 修改绘图逻辑，添加数据验证
    if y_flat.shape[0] == 0 or pred_flat.shape[0] == 0:
        raise ValueError("检测到空数据，无法绘制图表")
    
    # 绘制完整时间序列对比
    plt.plot(y_flat, label='Actual')  # 真值
    plt.plot(pred_flat, label=f'{args.model} Predicted')  # 预测
    
    # 添加垂直分割线显示每个预测窗口
    window_size = args.predict_days
    for i in range(1, len(y_test_original)):
        plt.axvline(x=i*window_size, color='gray', linestyle='--', alpha=0.3)
    
    plt.title(f'{args.model} {args.predict_days}-Day Forecast: Actual vs Predicted')
    plt.xlabel('Time Steps (each window={window_size} days)'.format(window_size=window_size))
    plt.ylabel('Global Active Power')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{args.model.lower()}_plot_{args.predict_days}.png'))
    plt.show()
