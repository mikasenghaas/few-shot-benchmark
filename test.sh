#!/bin/bash

: '
Run minimal configuration of experiments locally.

Runs all methods and datasets for a single epoch on CPU
Used mostly for debugging and testing.
'

# Experiment Parameters
exp_name=test
exp_device=cpu
exp_val_freq=100 # High to avoid running validation
exp_save_freq=100 # High to avoid saving too many checkpoints
exp_log_level=INFO # Low to avoid too many logs

# Dataset Parameters
dataset_subset=0.5 # No need to load data in parallel
dataset_loader_num_workers=0 # No need to load data in parallel
dataset_loader_pin_memory=false # No need to load data in parallel

# Training Parameters
train_stop_epoch=1 # Low to avoid running for too long

# Evaluation Parameters
eval_splits=[test]
# Logging parameters
wandb_mode=disabled

# Hyper-parameter grid
methods=( "baseline" "baseline_pp" "matchingnet" "protonet" "maml" )
datasets=( "swissprot" "tabula_muris" )

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
           dataset.subset=$dataset_subset \
           dataset.loader.num_workers=$dataset_loader_num_workers \
           dataset.loader.pin_memory=$dataset_loader_pin_memory \
           train.stop_epoch=$train_stop_epoch \
           eval.splits=$eval_splits \
           wandb.mode=$wandb_mode
    done
done