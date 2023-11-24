#!/bin/bash

: '
Example bash script for running experiments locally.

Runs all methods and datasets for 1 epoch each on CPU
each. This is useful for testing that the code runs
as expected.
'

# Experiment Parameters
exp_name=local-test
device=cpu
wandb_mode=disabled
eval_split=[test]
val_freq=100 # High to avoid running validation
save_freq=100 # High to avoid saving too many checkpoints
stop_epoch=1 # Low to avoid running for too long
log_level=CRITICAL

# Hyper-parameter grid
methods=( "baseline" "baseline_pp" "matchingnet" "protonet" "maml" )
datasets=( "swissprot" )

for dataset in "${datasets[@]}"
do
    for method in "${methods[@]}"
    do
        python run.py \
           exp.name=$exp_name \
           exp.device=$device \
           exp.val_freq=$val_freq \
           exp.save_freq=$save_freq \
           exp.log_level=$log_level \
           method=$method \
           method.stop_epoch=$stop_epoch \
           dataset=$dataset \
           eval_split=$eval_split \
           wandb.mode=$wandb_mode
    done
done