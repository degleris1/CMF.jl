#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G

ml julia/1.0.0
ml viz
ml py-matplotlib
srun julia synthetic_comparison.jl
