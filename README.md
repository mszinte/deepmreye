# DeepMReye
## About
---
We first proceed to a calibration of DeepMReye (deepmreyecalib) and next next </br>
evaluate it in condition of darkness and eye closed (deepmreyeclosed). </br>
This repository contains all code allowing us to analyse our dataset [OpenNeuro:DSXXXXX](https://openneuro.org/datasets/dsXXXX).</br>

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

#### DeepMReye decoding 
##### Calibration task 
- [x] run pretraining model calibration task [run_pretrain_calib.sh] (training_code/run_pretraining_sh.sh)
- [x] run fine-tuning model calibration task [run_training_calib.sh] (training_code/run_training_calib_sh.sh)
- [x] generate scaled model [generate_scaled_prediction.py] (training_code/generate_scaled_prediction.py)
- [x] evaluate performance (EE and corr) [calculate_model_measures_calib.py] (training_code/calculate_model_measures_calib.py)
- [x] figures [final_figures_calib.ipynb] (training_code/final_plots_calib)

##### Eye state tracking task
- [x] run pretraining model eye state tracking task [run_pretrain_closed.sh] (training_code/run_pretraining_closed_sh.sh)
- [x] run fine-tuning model eye state tracking task [run_training_calib.sh] (training_code/run_training_closed_sh.sh)
- [x] evaluate performance and figures: logistic regression classifier [new_classifier.ipynb] (training_code/new_classifier.ipynb)

##### Eyelid state decoding task 
- [x] run pretraining model eye state tracking task [run_pretraining_closed_vs_open_sh.sh] (training_code/run_pretraining_closed_vs_open_sh.sh)
- [x] run fine-tuning model eye state tracking task [run_training_closed_vs_open_sh.sh] (training_code/run_training_closed_vs_open_sh.sh)
- [x] evaluate performance and figures [eyes_closed_class_analysis.ipynb] (training_code/eyes_closed_class_analysis.ipynb)
