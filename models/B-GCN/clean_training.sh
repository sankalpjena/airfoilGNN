#!/bin/bash

# Removes checkpoints, and logs
rm -r *_checkpoints
rm -r *_csv_logs
rm -r *_tb_logs
rm -r split_dataset_info_seed_*
rm training_info_*.csv

# Removes post processing
rm -r post_train_data