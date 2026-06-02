#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --partition=THIN
#SBATCH --job-name=parallel_postproc
#SBATCH --output=logs/parallel_postproc_%j.out
#SBATCH -A dssc

# **********************************************************
# preprocess_parallel.sh
#
# This script is needed to request more resources to run 
# a parallel preprocessing of the corpus
#
# Run this script with 
#
# sbatch preprocess_parallel.sh "$(pwd)/lsi.cfg"
#
# **********************************************************

set -e

# --- CONFIG ---

config=$1
source $config

# Ensure joblib uses the allocated cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none

# --- MODULES ---

eval "$(/u/dssc/mzampar/miniconda3/bin/conda shell.bash hook)"
conda activate base

# --- MAIN ---

python -u preprocess_parallel.py $data_path $data_path_processed || { rm -r $data_path_processed ; echo "ERROR" ; exit 1; }

