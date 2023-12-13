#!/bin/bash

: '
Run all experiments for fixedd n-way and n-shot.
'

# Experiment Parameters (with hyperparameter grid for method and dataset)
group=new-tuned-benchmark
n_way=5
n_shot=5
methods=( "baseline" )
datasets=( "tabula_muris" )

lrs=(0.0001)
feat_dim=(64 512 1024)
sot_reg=(0.1)
distance=("cosine") 


for lr in "${lrs[@]}"
do
    for dim in "${feat_dim[@]}"
    do
        for reg in "${sot_reg[@]}"
        do
            for dist in "${distance[@]}"
            do
                for dataset in "${datasets[@]}"
                do
                    for method in "${methods[@]}"
                    do
                        python run.py \
                        group=$group \
                        method=$method \
                        dataset=$dataset \
                        use_sot=true \
                        n_way=$n_way \
                        n_shot=$n_shot \
                        train.lr=$lr \
                        dataset.backbone.feat_dim=$dim \
                        sot.ot_reg=$reg \
                        sot.distance=$dist
                    done
                done
            done
        done
    done
done
