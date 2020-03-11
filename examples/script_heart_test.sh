#!/bin/bash

for n in 1 32 64 128
do
for mid in K10_L30 K20_L15
do
for beta in 0.1 0.5 1.0
do
    echo $n $mid $beta
    motif=train_${n}_${mid}_B${beta}_results.mat
    filename=/home/asd/data/heart/results/${motif}
    if test -f "$filename"; then
        echo "go"
        julia script_heart_test.jl $motif testusers_${n}.mat
    else
        echo "pass"
    fi
done
done
done