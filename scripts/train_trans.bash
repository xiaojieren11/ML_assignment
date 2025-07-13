#!/bin/bash
GPU_ID=4
for _ in {1..5}
do
  CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --model Transformer \
                --num_epochs 100 \
                --predict_days 365 
done

for _ in {1..5}
do
  CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --model Transformer \
                --num_epochs 100 \
                --predict_days 90 
done