#!/bin/bash

python train.py --model Transformer \
                --num_epochs 10 \
                --predict_days 90 \
                --win_width 30 \