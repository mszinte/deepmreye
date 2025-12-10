#!/bin/bash
#SBATCH -p volta
#SBATCH -A b327
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=1:00:00
#SBATCH -e /scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_closed/log_outputs/deepmreye_%N_%j_%a.err
#SBATCH -o /scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_closed/log_outputs/deepmreye_%N_%j_%a.out
#SBATCH -J gpu_deepmreyeClosed
 
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python pretrain_closed.py /scratch/mszinte/data deepmreye 327