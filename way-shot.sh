#!/bin/bash

: '
Run n-way/ n-shot experiments for all fixed method and
experiment with and without SOT.
'

# Experiment Parameters (with hyperparameter grid for method and dataset)
group=new-way-shot
n_ways=( 10  8  6  4  2 )
n_shots=( 20  15  10  5  1 )
method="protonet"
dataset="tabula_muris"

for n_way in "${n_ways[@]}"
do
    for n_shot in "${n_shots[@]}"
    do
        python run.py \
        group=$group \
        method=$method \
        dataset=$dataset \
        use_sot=true \
        n_way=$n_way \
        n_shot=$n_shot \
        train.lr=0.1 \
        dataset.backbone.feat_dim=1024 \
        sot.ot_reg=0.01 \
        sot.distance=cosine
    done
done
