#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=200:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G

ml julia/1.0.0
ml viz
ml py-matplotlib
srun julia birdsong.jl
