#!/bin/bash

python train.py --model LSTM \
                --eval_only \
                --predict_days 90 \
                --model_path ./weight/lstm_best_model_90.pth 