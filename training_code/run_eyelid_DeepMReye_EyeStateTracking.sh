#!/bin/bash
#SBATCH -p volta
#SBATCH -A b327
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --time=20:00:00
#SBATCH -e /scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_eyelid_state/log_outputs/eyelid_DeepMReye_EyeStateTracking_%N_%j.err
#SBATCH -o /scratch/mszinte/data/deepmreye/derivatives/int_deepmreye/deepmreye_eyelid_state/log_outputs/eyelid_DeepMReye_EyeStateTracking_%N_%j.out
#SBATCH -J gpu_eyelid_DeepMReye_EST

export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python train_eyelid_state.py \
    /scratch/mszinte/data \
    deepmreye \
    eyestate \
    eyesclosed \
    modelinference_gaze_DeepMReyeClosed.h5 \
    DeepMReye_EyeStateTracking
