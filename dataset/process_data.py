import pandas as pd
import numpy as np

def process_data(train_file, test_file, output_train_file, output_test_file):
    """
    处理 train.csv 和 test.csv 数据集，按天分组并计算指定列的总和与平均值。
    """

    # 定义需要计算总和与平均值的列
    sum_columns = ['Global_active_power', 'Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2','Sub_metering_3']
    mean_columns = ['Voltage', 'Global_intensity']
    first_columns = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

    # 处理训练数据
    train_df = pd.read_csv(train_file)
    train_df['DateTime'] = pd.to_datetime(train_df['DateTime'], format='%Y-%m-%d %H:%M:%S').dt.date

    # 将相关列转换为数值类型
    cols_to_convert = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in cols_to_convert:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

    # 填充缺失值
    for col in cols_to_convert:
        train_df[col] = train_df[col].fillna(train_df[col].mean())

    # 计算 sub_metering_remainder
    train_df['Sub_metering_3'] = train_df['Sub_metering_3'].fillna(0)
    train_df['Sub_metering_remainder'] = (train_df['Global_active_power'] * 1000 / 60) - (train_df['Sub_metering_1'] + train_df['Sub_metering_2'] + train_df['Sub_metering_3'])
    sum_columns.append('Sub_metering_remainder')

    print(train_df.dtypes)
    train_grouped = train_df.groupby('DateTime').agg(
        **{col: pd.NamedAgg(column=col, aggfunc='sum') for col in sum_columns},
        **{col: pd.NamedAgg(column=col, aggfunc='mean') for col in mean_columns},
        **{col: pd.NamedAgg(column=col, aggfunc='first') for col in first_columns}
    ).reset_index()
    train_grouped.to_csv(output_train_file, index=False)

    # 处理测试数据
    # 定义列名
    columns = ['DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

    # 创建包含列名的 DataFrame
    columns_df = pd.DataFrame([columns])

    # 读取原始测试数据
    test_df = pd.read_csv(test_file, header=None)

    # 将列名添加到测试数据的最前面
    test_df = pd.concat([columns_df, test_df], ignore_index=True)

    # 将更新后的数据保存到临时文件
    temp_test_file = 'test_temp.csv'
    test_df.to_csv(temp_test_file, index=False, header=False)

    # 从临时文件读取数据
    test_df = pd.read_csv(temp_test_file)

    test_df['DateTime'] = pd.to_datetime(test_df['DateTime'], format='%Y-%m-%d %H:%M:%S').dt.date

    # 将相关列转换为数值类型
    for col in cols_to_convert:
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')

    # 填充缺失值
    for col in cols_to_convert:
        test_df[col] = test_df[col].fillna(test_df[col].mean())

    # 计算 sub_metering_remainder
    test_df['Sub_metering_3'] = test_df['Sub_metering_3'].fillna(0)
    test_df['Sub_metering_remainder'] = (test_df['Global_active_power'] * 1000 / 60) - (test_df['Sub_metering_1'] + test_df['Sub_metering_2'] + test_df['Sub_metering_3'])
    
    test_grouped = test_df.groupby('DateTime').agg(
        **{col: pd.NamedAgg(column=col, aggfunc='sum') for col in sum_columns},
        **{col: pd.NamedAgg(column=col, aggfunc='mean') for col in mean_columns},
        **{col: pd.NamedAgg(column=col, aggfunc='first') for col in first_columns}
    ).reset_index()
    test_grouped.to_csv(output_test_file, index=False)

if __name__ == "__main__":
    # 定义输入和输出文件路径
    train_file = 'train.csv'
    test_file = 'test.csv'
    output_train_file = 'train_processed.csv'
    output_test_file = 'test_processed.csv'

    # 调用数据处理函数
    process_data(train_file, test_file, output_train_file, output_test_file)
