# Copy tar files from XNAT to sourcedata folder
# untar the file
tar vfxz XNAT-DeepMReye_NIFTI-2024-03-25_10h52.tar.gz -C /scratch/mszinte/data/deepmreye/
# delete the newly created xnat subfolder and move data to root

# copy code folder from another project
rsync -avuz /scratch/mszinte/data/RetinoMaps/code/ /scratch/mszinte/data/deepmreye/code/

# Change permissions to all data
chmod -Rf 771 /scratch/mszinte/data/deepmreye
chgrp -Rf 327 /scratch/mszinte/data/deepmreye