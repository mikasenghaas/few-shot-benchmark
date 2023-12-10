#!/bin/bash

: '
Run experiment to understand the effect of SOT with
and without further embedding modules.
'

# Experiment Parameters (with hyperparameter grid for method and dataset)
group=sot-interaction
sot=( false true ) 
embed_supports=( false true )
embed_queries=( false true )
n_way=5
n_shot=5
method="matchingnet"
dataset="swissprot"

for embed_support in "${embed_supports[@]}"
do
    for embed_query in "${embed_queries[@]}"
    do
        for use_sot in "${sot[@]}"
        do
            python run.py \
            name=sot-$use_sot-support-$embed_support-queries-$embed_query \
            group=$group \
            method=$method \
            dataset=$dataset \
            use_sot=$use_sot \
            n_way=$n_way \
            n_shot=$n_shot \
            method.embed_support=$embed_support \
            method.embed_query=$embed_query 
        done
    done
done
