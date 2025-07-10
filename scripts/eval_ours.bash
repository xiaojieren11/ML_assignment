#!/bin/bash

python train.py --model Ours \
                --eval_only \
                --predict_days 365 \
                --model_path ./weight/ours_best_model_365.pth 