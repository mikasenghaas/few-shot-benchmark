#!/bin/bash

: '
Run experiments on GCP

Runs all methods and datasets for a single epoch on CPU
Used mostly for debugging and testing.
'

# Experiment Parameters
exp_name=gcp
exp_device=cuda
exp_val_freq=1 # High to avoid running validation
exp_save_freq=10 # High to avoid saving too many checkpoints
exp_log_level=CRITICAL # Low to avoid too many logs
wandb_mode=online

# Dataset Parameters
dataset_loader_num_workers=0 # No need to load data in parallel
dataset_loader_pin_memory=false # No need to load data in parallel

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
           dataset.loader.num_workers=$dataset_loader_num_workers \
           dataset.loader.pin_memory=$dataset_loader_pin_memory \
           wandb.mode=$wandb_mode
    done
done