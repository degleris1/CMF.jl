#!/bin/bash

#SBATCH -t 08:00:00
#SBATCH -J heart1.0
#SBATCH --mem=16G
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=degleris@stanford.edu

# load julia
module load julia

# run jobs
echo "Starting at" `date`
script=/home/degleris/CMF.jl/examples/script_heart.jl

for beta in 0.1 0.5 1
do
for file in train_1 train_32 train_64 train_128
do
	julia $script $file 10 30 $beta &
    	julia $script $file 20 15 $beta &
done
	wait
	echo "Finished" $file "at" `date`
done
