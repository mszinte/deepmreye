# DeepMReye
## About
---
We first proceed to a calibration of DeepMReye (deepmreyecalib )and next next </br>
evaluate it in condition of darkness and eye closed (deepmreyeclosed)
This repository contain all code allowing us to analyse our dataset [OpenNeuro:DSXXXXX](https://openneuro.org/datasets/dsXXXX).</br>

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
[Inkscape](https://inkscape.org/);
[workbench](https://humanconnectome.org/software/connectome-workbench)
</br>


## **Data analysis**
---

### To Do 

## Pre-processing
---
#### BIDS
- [x] Copy brain data from XNAT [copy_data.py](analysis_code/preproc/bids/bids_copy_data.sh)
- [ ] Copy behavioral data [copy_data.py](analysis_code/preproc/bids/bids_copy_data.sh) 
- [ ] Rename sessions to add behaviour data [rename-ses.sh] (analysis_code/preproc/bids/rename_ses.sh)
- [ ] Deface participants t1w image [deface_sbatch.py](analysis_code/preproc/bids/deface_sbatch.py) 
    </br>Note: run script for each subject separately.
- [ ] put eyetracking in latest BEP020 BIDS format 
- [ ] Correct BIDS problems [correct_bids_files.ipynb](analysis_code/preproc/bids/correct_bids_files.ipynb)
- [ ] Validate bids format [https://bids-standard.github.io/bids-validator/] / alternately, use a docker [https://pypi.org/project/bids-validator/]
    </br>Note: for the webpage, use FireFox and wait for at least 30 min, even if nothing seems to happen.

#### Structural preprocessing
- [ ] fMRIprep with anat-only option [fmriprep_sbatch.py](analysis_code/preproc/functional/fmriprep_sbatch.py)
- [ ] Create sagital view video before manual edit [sagital_view.py](analysis_code/preproc/anatomical/sagital_view.py)
- [ ] Manual edit of brain segmentation [pial_edits.sh](analysis_code/preproc/anatomical/pial_edits.sh)
- [ ] FreeSurfer with new brainmask manually edited [freesurfer_pial.py](analysis_code/preproc/anatomical/freesurfer_pial.py)
- [ ] Create sagital view video before after edit [sagital_view.py](analysis_code/preproc/anatomical/sagital_view.py)
- [ ] Make cut in the brains for flattening [cortex_cuts.sh](analysis_code/preproc/anatomical/cortex_cuts.sh)
- [ ] Flatten the cut brains [flatten_sbatch.py](analysis_code/preproc/anatomical/flatten_sbatch.py)

#### Functional preprocessing
- [ ] fMRIprep [fmriprep_sbatch.py](analysis_code/preproc/functional/fmriprep_sbatch.py)
- [ ] Load freesurfer and import subject in pycortex db [freesurfer_import_pycortex.py](analysis_code/preproc/functional/freesurfer_import_pycortex.py)
- [ ] High-pass, z-score, run correlations, average and leave-one-out average [preproc_end_sbatch.py](analysis_code/preproc/functional/preproc_end_sbatch.py) 
- [ ] Make timeseries inter-run correlation maps with pycortex [pycortex_corr_maps.py](analysis_code/preproc/functional/pycortex_corr_maps.py)

#### Eyetracking analysis
- [ ] To make