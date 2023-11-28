#!/bin/bash

: '
Run minimal configuration of experiments locally.

Runs all methods and datasets for a single epoch on CPU
Used mostly for debugging and testing.
'

# Experiment Parameters
exp_name=local
exp_device=cpu
exp_val_freq=1 # High to avoid running validation
exp_save_freq=10 # High to avoid saving too many checkpoints
exp_log_level=INFO # Low to avoid too many logs

# Dataset Parameters
dataset_loader_num_workers=0 # No need to load data in parallel
dataset_loader_pin_memory=false # No need to load data in parallel

# Training Parameters
train_stop_epoch=10 # Train all models for 10 epochs

# Hyper-parameter grid
methods=( "matchingnet" "protonet" "maml" )
datasets=( "tabula_muris" )

for dataset in "${datasets[@]}"
do
    for method in "${methods[@]}"
    do
        python run.py \
           method=$method \
           dataset=$dataset \
           exp.name=$exp_name \
           exp.device=$exp_device \
           exp.val_freq=$exp_val_freq \
           exp.save_freq=$exp_save_freq \
           exp.log_level=$exp_log_level \
           dataset.loader.num_workers=$dataset_loader_num_workers \
           dataset.loader.pin_memory=$dataset_loader_pin_memory \
           train.stop_epoch=$train_stop_epoch
    done
done