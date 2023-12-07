#!/bin/bash

: '
Run all experiments for fixedd n-way and n-shot.
'

# Experiment Parameters (with hyperparameter grid for method and dataset)
group=benchmark
n_way=5
n_shot=5
sot=( false true ) 
methods=( "matchingnet" "protonet" "maml" "baseline" "baseline_pp" )
datasets=( "swissprot" "tabula_muris" )

for dataset in "${datasets[@]}"
do
    for method in "${methods[@]}"
    do
        for use_sot in "${sot[@]}"
        do
            python run.py \
            group=$group \
            method=$method \
            dataset=$dataset \
            use_sot=$use_sot \
            n_way=$n_way \
            n_shot=$n_shot
        done
    done
done
