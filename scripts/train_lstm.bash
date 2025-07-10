#!/bin/bash

for _ in {1..5}
do
  python train.py --model LSTM \
                --num_epochs 200 \
                --predict_days 365 
done

for _ in {1..5}
do
  python train.py --model LSTM \
                --num_epochs 200 \
                --predict_days 90 
done