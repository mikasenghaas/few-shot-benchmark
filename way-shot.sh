#!/bin/bash

: '
Run n-way/ n-shot experiments for all fixed method and
experiment with and without SOT.
'

# Experiment Parameters (with hyperparameter grid for method and dataset)
group=tuned-way-shot
n_ways=( 10  8  6  4  2 )
n_shots=( 20  15  10  5  1 )
sot=( false true ) 
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
          use_sot=false \
          n_way=$n_way \
          n_shot=$n_shot

          python run.py -m \
          group=$group \
          method=$method \
          dataset=$dataset \
          use_sot=true \
          n_way=$n_way \
          n_shot=$n_shot
    done
done
