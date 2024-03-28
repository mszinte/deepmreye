# Copy tar files from XNAT to sourcedata folder
# untar the file
tar vfxz XNAT-DeepMReye_NIFTI-2024-03-25_10h52.tar.gz -C /scratch/mszinte/data/deepmreye/
# delete the newly created xnat subfolder and move data to root

# copy code folder from another project
rsync -avuz /scratch/mszinte/data/RetinoMaps/code/ /scratch/mszinte/data/deepmreye/code/

# Change permissions to all data
chmod -Rf 771 /scratch/mszinte/data/deepmreye
chgrp -Rf 327 /scratch/mszinte/data/deepmreye

# mount locally & on invibe server by adding to .bash_profile :
#alias mdeep='sshfs -p 8822 skling@login.mesocentre.univ-amu.fr:/scratch/mszinte/data/deepmreye ~/disks/meso_shared/deepmreye -olocal,auto_cache,reconnect,defer_permissions,noappledouble,negative_vncache,volname=deepmreye'
#alias udeep='umount ~/disks/meso_shared/deepmreye/'

# manually add participants.tsv, participants.json, tasks.json, README, dataset_description.json to folder 

# copy eye tracking data (Training session)
# mkdir /scratch/mszinte/data/deepmreye/sub-01/ses-01/func
# mkdir /scratch/mszinte/data/deepmreye/sub-02/ses-01/func

# rename data accordingly first using rename_ses.sh

scp -r -P 8822 /Users/sinakling/projects/deepmreyecalib/experiment_code/data/sub-01/ses-01/beh skling@login.mesocentre.univ-amu.fr:/scratch/mszinte/data/deepmreye/sub-01/ses-01/behav
scp -r -P 8822 /Users/sinakling/projects/deepmreyecalib/experiment_code/data/sub-02/ses-01/beh skling@login.mesocentre.univ-amu.fr:/scratch/mszinte/data/deepmreye/sub-02/ses-01/behav

scp -r -P 8822 /Users/sinakling/projects/deepmreyeclosed/data/sub-01/ses-01/beh skling@login.mesocentre.univ-amu.fr:/scratch/mszinte/data/deepmreye/sub-01/ses-01/behav
scp -r -P 8822 /Users/sinakling/projects/deepmreyeclosed/data/sub-02/ses-01/beh skling@login.mesocentre.univ-amu.fr:/scratch/mszinte/data/deepmreye/sub-02/ses-01/behav

# remove behav folder and move data to the root 


# copy eyetracking data (Functional session)

scp -r -P 8822 /Users/sinakling/projects/deepmreyecalib/experiment_code/data/sub-01/ses-01/func skling@login.mesocentre.univ-amu.fr:/scratch/mszinte/data/deepmreye/sub-01/ses-02/behav
scp -r -P 8822 /Users/sinakling/projects/deepmreyecalib/experiment_code/data/sub-02/ses-01/func skling@login.mesocentre.univ-amu.fr:/scratch/mszinte/data/deepmreye/sub-02/ses-02/behav

scp -r -P 8822 /Users/sinakling/projects/deepmreyeclosed/data/sub-01/ses-01/func skling@login.mesocentre.univ-amu.fr:/scratch/mszinte/data/deepmreye/sub-01/ses-02/behav
scp -r -P 8822 /Users/sinakling/projects/deepmreyeclosed/data/sub-02/ses-01/func skling@login.mesocentre.univ-amu.fr:/scratch/mszinte/data/deepmreye/sub-02/ses-02/behav

# remove behav folder and move data to the root 

# add .bidsignore file to data folder 
