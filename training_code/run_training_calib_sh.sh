#!/bin/bash
#SBATCH -p volta
#SBATCH -A b327
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=15:00:00
#SBATCH -e /scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calib/log_outputs/deepmreye_%N_%j_%a.err
#SBATCH -o /scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_calib/log_outputs/deepmreye_%N_%j_%a.out
#SBATCH -J gpu_deepmreyeCalib
 
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python train_calib.py /scratch/mszinte/data deepmreye 327