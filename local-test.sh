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
stop_epoch=1
eval_split=[test]
wandb_mode=disabled

# Hyper-parameter grid
methods=( "baseline" "baseline_pp" "matchingnet" "protonet" "maml" )
datasets=( "tabula_muris" "swissprot" )

for dataset in "${datasets[@]}"
do
    for method in "${methods[@]}"
    do
        python run.py \
           exp.name=$exp_name \
           exp.device=$device \
           method=$method \
           method.stop_epoch=$stop_epoch \
           dataset=$dataset \
           eval_split=$eval_split \
           wandb.mode=$wandb_mode 
    done
done