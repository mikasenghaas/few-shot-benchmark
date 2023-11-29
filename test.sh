#!/bin/bash

: '
Run minimal configuration of experiments locally.

Runs all methods and datasets for a single epoch on CPU
Used mostly for debugging and testing.
'

# Experiment Parameters (with hyperparameter grid for method and dataset)
group=test
n_way=5
n_shot=5
sot=false
methods=( "baseline" "baseline_pp" "matchingnet" "protonet" "maml" )
datasets=( "swissprot" "tabula_muris" )

for dataset in "${datasets[@]}"
do
    for method in "${methods[@]}"
    do
        python run.py -m \
           group=$group \
           method=$method \
           dataset=$dataset \
           sot=$sot \
           n_way=$n_way \
           n_shot=$n_shot \
           train.max_epochs=1 \
           eval.splits=[test] \
           wandb.mode=disabled
    done
done