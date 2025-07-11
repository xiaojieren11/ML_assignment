import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="Train LSTM and Transformer models for power prediction.")

    parser.add_argument('--win_width', type=int, default=7, help='Number of time steps for the sliding window.')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for the LSTM model.')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension for the Transformer model.')
    parser.add_argument('--dense_dim', type=int, default=64, help='Dense dimension for the Transformer model.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads for the Transformer model.')
    parser.add_argument('--output_size', type=int, default=1, help='Output size for the models.')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--train_size', type=int, default=90, help='Size of the training dataset (last N days).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizers.')
    parser.add_argument('--eval_only', action='store_true', help='if true, only evaluate the model')
    parser.add_argument('--model_path', type=str, help='Path to the saved model for evaluation.')
    parser.add_argument('--model', type=str, default='LSTM', help='choose from (LSTM, Transformer,Ours)')
    parser.add_argument('--predict_days', type=int, choices=[90, 365], default=90, help='days to predict (90 or 365).')
    parser.add_argument('--train_days', type=int, default=90, help='days for training.')

    args = parser.parse_args()
    return args
    