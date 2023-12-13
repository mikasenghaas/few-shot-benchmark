#!/bin/bash

: '
Run all experiments for fixedd n-way and n-shot.
'

# Experiment Parameters (with hyperparameter grid for method and dataset)
group=new-tuned-benchmark
n_way=5
n_shot=5
methods=( "baseline_pp" )
datasets=( "swissprot" ) 

for dataset in "${datasets[@]}"
do
    for method in "${methods[@]}"
    do
          python run.py -m \
          group=$group \
          method=$method \
          dataset=$dataset \
          use_sot=true \
          n_way=$n_way \
          n_shot=$n_shot
    done
done
