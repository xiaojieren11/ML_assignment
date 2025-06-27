# 超参数配置
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Train LSTM and Transformer models for power prediction.")

    parser.add_argument('--time_steps', type=int, default=30, help='Number of time steps for the sliding window.')
    parser.add_argument('--hidden_size', type=int, default=32, help='Hidden size for the LSTM model.')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension for the Transformer model.')
    parser.add_argument('--dense_dim', type=int, default=16, help='Dense dimension for the Transformer model.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for the Transformer model.')
    parser.add_argument('--output_size', type=int, default=1, help='Output size for the models.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--train_size', type=int, default=90, help='Size of the training dataset (last N days).')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizers.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the Transformer model.')
    parser.add_argument('--eval_only', action='store_true', help='是否加载模型')
    parser.add_argument('--model_path', type=str, help='模型路径')
    parser.add_argument('--model', type=str, default='LSTM', help='选择模型 (LSTM, Transformer)')
    parser.add_argument('--predict_days', type=int, default=90, help='预测天数')

    args = parser.parse_args()
    return args