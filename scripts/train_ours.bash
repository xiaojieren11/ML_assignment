#!/bin/bash
GPU_ID=7
CUDA_VISIBLE_DEVICES=4
for _ in {1..5}
do
  CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --model Ours \
                --num_epochs 100 \
                --predict_days 365 
done

for _ in {1..5}
do
  CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --model Ours \
                --num_epochs 100 \
                --predict_days 90 
done