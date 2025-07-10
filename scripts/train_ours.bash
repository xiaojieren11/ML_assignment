#!/bin/bash

for _ in {1..5}
do
  python train.py --model Ours \
                --num_epochs 200 \
                --predict_days 365 \
                --win_width 30 
done

for _ in {1..5}
do
  python train.py --model Ours \
                --num_epochs 200 \
                --predict_days 90 \
                --win_width 30 
done