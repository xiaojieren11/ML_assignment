#!/bin/bash

cd ../
python train.py --model LSTM --eval_only --model_path ./weight/lstm_best_model.pth --predict_days 90