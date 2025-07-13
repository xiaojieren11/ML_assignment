#!/bin/bash

python ./mean_std.py --file_path ./output/lstm_results_90/evaluation_metrics.csv
python ./mean_std.py --file_path ./output/lstm_results_365/evaluation_metrics.csv

python ./mean_std.py --file_path ./output/transformer_results_90/evaluation_metrics.csv
python ./mean_std.py --file_path ./output/transformer_results_365/evaluation_metrics.csv

python ./mean_std.py --file_path ./output/ours_results_90/evaluation_metrics.csv
python ./mean_std.py --file_path ./output/ours_results_365/evaluation_metrics.csv