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
eval_split=["test"]
wandb_mode=offline

# Hyper-parameter grid
methods=( "baseline" "baseline_pp" "matchingnet" "protonet" "maml" )
datasets=( "swissprot" "tabula_muris" )

for method in "${methods[@]}"
do
    for dataset in "${datasets[@]}"
    do
        python run.py \
           exp.name=$exp_name \
           device=$device \
           method=$method \
           method.stop_epoch=$stop_epoch \
           dataset=$dataset \
           dataset.eval_split=$eval_split \
           wandb.mode=$wand_mode 
    done
done