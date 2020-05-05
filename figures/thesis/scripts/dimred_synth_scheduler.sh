#!/bin/bash

savepath=/farmshare/user_data/degleris/dim/
script=/home/degleris/CMF.jl/figures/thesis/scripts/dimred_synth_script.sh
for K in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
    # run worst case bounds
    bash script K 1 savepath

    # run actual model
    bash script K 30 savepath
done

# run best case bounds
for K in 15 30 45 60 75 90 105 120 135 150 165 180 195 210 225
do
    bash script K 1 savepath
done

echo "Scheduled."