import pandas as pd
import argparse

# 设置参数解析器
parser = argparse.ArgumentParser(description='处理评估指标文件并添加平均值与标准差')
parser.add_argument('--file_path', type=str, required=True, help='CSV 文件路径')

args = parser.parse_args()

# 读取原始数据
df = pd.read_csv(args.file_path)

# 计算平均值和标准差
stats = df.groupby('Model')[['MSE', 'MAE']].agg(['mean', 'std'])

# 构建新行
new_rows = []
for model in stats.index:
    new_row = {
        'Model': model,
        'Predict_Days': 'Average',
        'MSE': stats.loc[model, ('MSE', 'mean')],
        'MAE': stats.loc[model, ('MAE', 'mean')],
        'MSE_std': stats.loc[model, ('MSE', 'std')],
        'MAE_std': stats.loc[model, ('MAE', 'std')]
    }
    new_rows.append(new_row)

new_df = pd.DataFrame(new_rows)

# 合并数据
updated_df = pd.concat([df, new_df], ignore_index=True)

# 写回文件
updated_df.to_csv(args.file_path, index=False)