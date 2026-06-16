# DeepMReye
## About
---
We first proceed to a calibration of DeepMReye (deepmreyecalib) and next next </br>
evaluate it in condition of darkness and eye closed (deepmreyeclosed). </br>
This repository contains all code allowing us to analyse our dataset [OpenNeuro:DS006833](https://openneuro.org/datasets/ds006833).</br>
All model weights are available here (https://figshare.com/s/fe874ffbb37f5bc08645).<br>
We also provide a ready to use code repository for using the model weights here (https://github.com/sinaklg/int_deepmreye/tree/main).<br>

---
## Authors (alphabetic order): 
---
Uriel LASCOMBES, Sina KLING, Guillaume MASSON, Matthias NAU, & Martin SZINTE

## Main dependencies
---
[PyDeface](https://github.com/poldracklab/pydeface); 
[fMRIprep](https://fmriprep.org/en/stable/); 
[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/);
[FFmpeg](https://ffmpeg.org/);
[FSL](https://fsl.fmrib.ox.ac.uk);
</br>


## **Data analysis**
---
## Pre-processing
---
#### BIDS
- [x] Copy brain data from XNAT [copy_data.py](analysis_code/preproc/bids/bids_copy_data.sh)
- [x] Copy behavioral data [copy_data.py](analysis_code/preproc/bids/bids_copy_data.sh) 
- [x] Rename sessions to add behaviour data [rename-ses.sh] (analysis_code/preproc/bids/rename_ses.sh)
- [x] Deface participants t1w image [deface_sbatch.py](analysis_code/preproc/bids/deface_sbatch.py) 
    </br>Note: defaced after main analysis 
- [x] put eyetracking in latest BEP020 BIDS format 
- [x] Validate bids format [https://bids-standard.github.io/bids-validator/] 

#### Structural preprocessing
- [x] fMRIprep with anat-only option [fmriprep_sbatch.py](analysis_code/preproc/functional/fmriprep_sbatch.py)

#### Functional preprocessing
- [x] fMRIprep [fmriprep_sbatch.py](analysis_code/preproc/functional/fmriprep_sbatch.py)

#### Eyetracking analysis
- [x] Run analysis code in deepmreyecalib and deepmreyeclosed repos 

#### DeepMReye Fine-tuning 
##### Visuomotor calibration tasks
- [x] generate eyetracking labels for calibration task [eyetracking_labels.py](training_code/eyetracking_preproc.py)
- [x] run pretraining model calibration task [run_pretrain_calib.sh](training_code/run_pretraining_sh.sh)
- [x] run fine-tuning model calibration task [run_training_calib.sh](training_code/run_training_calib_sh.sh)

- [X] generate scaled model predictions [generate_scaled_prediction.py](/training_code/generate_scaled_prediction.py)
- [X] generate target position labels [generate_simulated_labels_calib.py](training_code/generate_simulated_labels_calib.py)

##### Eye/Vision state tasks
- [x] generate eyetracking labels for eye/vision state task [eyetracking_labels.py](training_code/eyetracking_preproc.py)
- [x] run pretraining model eye/vision state task [run_pretrain_closed.sh](training_code/run_pretraining_closed_sh.sh)
- [x] run fine-tuning model eye/vision state task [run_training_calib.sh](training_code/run_training_closed_sh.sh)

##### Eyelid state decoding task 
- [X] run fine-tuning with pretrained DeepMReye weights [run_eyelid_DeepMReye.sh](training_code/run_eyelid_DeepMReye.sh)
- [X] run fine-tuning with calibration tasks weights [run_eyelid_DeepMReye_Calibration.sh](training_code/run_eyelid_DeepMReye_Calibration.sh)
- [X] run fine-tuning with eye/vision state tasks weights [run_eyelid_DeepMReye_EyeStateTracking.sh](training_code/run_eyelid_DeepMReye_EyeStatetracking.sh)

#### Decoding performance analysis 
##### Visuomotor calibration tasks
- [x] calculate correlation for calibration fine tuning [calculate_corr_calib.py](analysis_code/postproc/calculate_corr_calib.py)
- [x] calculate EE for calibration fine tuning [calculate_ee_calib.py](analysis_code/postproc/calculate_ee_calib.py)
- [x] figures [final_figures_calib.ipynb](analysisi_code_code/postproc/final_plots_calib.ipynb)
- [x] results overview in: [results_calib.ipynb](analysis_code/postproc/results_calib.ipynb)

##### Eye/Vision state tasks
- [x] evaluate performance and figures: logistic regression classifier [classifier.py](analysis_code/postproc/classifier.py)
- [x] figures [classifier_plots.ipynb](analysis_code/postproc/classifier_plots.ipynb)
- [x] results overview in: [results_closed.ipynb](analysis_code/postproc/results_closed.ipynb)

##### Eyelid state decoding task 
- [x] evaluate performance and figures [eyes_closed_class_analysis.ipynb](analysis_code/postproc/eyes_closed_class_analysis.ipynb)
