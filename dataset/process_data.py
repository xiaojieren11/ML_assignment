import pandas as pd
import numpy as np

def preprocess_data(file_path):
    # 13个列名
    column_names = [
        'DateTime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
    ]

    try:
        if 'train.csv' in file_path:
            # train.csv 有一个无效的标题行，跳过并使用正确的13个列名
            df = pd.read_csv(
                file_path,
                sep=',',
                header=None,
                skiprows=1,
                names=column_names,
                low_memory=False,
                on_bad_lines='skip'
            )
        elif 'test.csv' in file_path:
            # test.csv 没有标题行，直接使用13个列名
            df = pd.read_csv(
                file_path,
                sep=',',
                header=None,
                names=column_names,
                low_memory=False,
                on_bad_lines='skip'
            )
        else:
            raise ValueError("无法识别的文件路径，请确保文件名为 train.csv 或 test.csv")
    except Exception as e:
        print(f"读取文件 {file_path} 时出现严重错误，请检查文件格式和内容。错误: {e}")
        raise

    print(f"文件 '{file_path}' 已根据统一格式加载。")

    df['datetime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    df = df.set_index('datetime')
    df = df.drop('DateTime', axis=1)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.fillna(method='ffill', inplace=True)

    if all(c in df.columns for c in ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']):
        df['sub_metering_remainder'] = (df['Global_active_power'] * 1000 / 60) - \
                                       (df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3'])
    # 定义聚合规则
    aggregation_rules = {
        'Global_active_power': 'sum', 'Global_reactive_power': 'sum', 'Voltage': 'mean',
        'Global_intensity': 'mean', 'Sub_metering_1': 'sum', 'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum', 'sub_metering_remainder': 'sum', 'RR': 'first',
        'NBJRR1': 'first', 'NBJRR5': 'first', 'NBJRR10': 'first', 'NBJBROU': 'first'
    }
    cols_to_agg = [col for col in aggregation_rules if col in df.columns]
    df_daily = df[cols_to_agg].resample('D').agg({k: aggregation_rules[k] for k in cols_to_agg})
    df_daily.fillna(method='ffill', inplace=True)

    if 'RR' in df_daily.columns:
        df_daily['RR'] = df_daily['RR'] / 10.0

    return df_daily

if __name__ == "__main__":
    # 定义输入和输出文件路径
    train_file = 'train.csv'
    test_file = 'test.csv'
    output_train_file = 'train_processed.csv'
    output_test_file = 'test_processed.csv'

    preprocess_data(train_file).to_csv(output_train_file)
    preprocess_data(test_file).to_csv(output_test_file)
