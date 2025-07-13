import torch
from models.LSTM import LSTMModel 
from models.Transformer import TransformerModel 
from models.Ours import CNNTransformer 
from train import create_model
from config import get_parser

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

args = get_parser()

# model_path = './weight/lstm_best_model_90.pth'
model_path = './weight/transformer_best_model_365.pth'
# model_path = './weight/ours_best_model_365.pth'
output_path = './models/model_info.txt'

model = create_model(
        model_type='Transformer',
        input_size=13,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        embed_dim=args.embed_dim,
        dense_dim=args.dense_dim,
        num_heads=args.num_heads,
)
model.load_state_dict(torch.load(model_path))
model.eval()
with open(output_path, 'a') as f:
    # 打印参数数量
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    f.write(f"Total trainable parameters: {total_params:,}\n")
    
    # 打印模型结构
    print(model)
    f.write(str(model) + "\n")