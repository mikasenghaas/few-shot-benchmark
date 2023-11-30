#!/bin/bash

: '
Run group of experiments on GCP VM.
'

# Experiment Parameters (with hyperparameter grid for method and dataset)
group=gcp
n_way=5
n_shot=5
sot=( false true ) 
methods=( "baseline" "baseline_pp" "matchingnet" "protonet" "maml" )
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
            n_shot=$n_shot \
            general.device=cuda
        done
    done
done
