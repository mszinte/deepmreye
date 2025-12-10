#!/bin/bash

# Check if the user provided a subject name
if [ -z "$1" ]; then
    echo "Please provide a subject name (e.g., sub-05)."
    exit 1
fi

SUBJECT=$1
REMOTE_PATH="skling@login.mesocentre.univ-amu.fr:/scratch/mszinte/data/deepmreye/derivatives/fmriprep/fmriprep"
LOCAL_BASE_PATH="~/projects/DeepMReye"
PORT=8822

# Create directories for the subject
mkdir -p "${LOCAL_BASE_PATH}/closed_data/functional_data/${SUBJECT}"
mkdir -p "${LOCAL_BASE_PATH}/calib_data/functional_data/${SUBJECT}"

# Function to generate and execute the scp command
execute_scp_command() {
    local task=$1
    local run=$2
    local folder=$3
    
    local scp_command="scp -P ${PORT} ${REMOTE_PATH}/${SUBJECT}/ses-02/func/${SUBJECT}_ses-02_task-${task}_run-${run}_space-T1w_desc-preproc_bold.nii.gz ${LOCAL_BASE_PATH}/${folder}/functional_data/${SUBJECT}/"
    echo "Executing: $scp_command"
    eval $scp_command
}

# Execute scp commands for DeepMReyeClosed
for run in 01 02 03; do
    execute_scp_command "DeepMReyeClosed" $run "closed_data"
done

# Execute scp commands for DeepMReyeCalib
for run in 01 02 03; do
    execute_scp_command "DeepMReyeCalib" $run "calib_data"
done
