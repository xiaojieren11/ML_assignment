#!/bin/bash

python train.py --model Ours1 \
                --eval_only \
                --predict_days 365 \
                --win_width 30 \
                --model_path ./weight/ours1_best_model_365.pth 