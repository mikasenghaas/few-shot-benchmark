#!/bin/bash

: '
Run experiment to understand the behaviour of
ProtoNet with and without SOT on SwissProt.
'

# Experiment Parameters
group=model-behaviour
sot=( false true ) 
n_way=5
n_shot=5
method="protonet"
dataset="swissprot"

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
