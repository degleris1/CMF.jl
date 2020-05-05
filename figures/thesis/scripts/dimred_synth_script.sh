#!/bin/bash

#SBATCH -t 03:00:00
#SBATCH -J dimsynth
#SBATCH --mem=8G
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=degleris@stanford.edu

# load julia
module load julia

# run jobs
echo "Starting at" `date`
script=/home/degleris/CMF.jl/figures/thesis/scripts/dimred_synth.jl
julia $script $1 $2 $3
wait
echo "Finished at" `date`