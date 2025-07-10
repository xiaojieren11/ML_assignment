#!/bin/bash

python train.py --model Transformer \
                --eval_only \
                --predict_days 365 \
                --model_path ./weight/transformer_best_model_365.pth