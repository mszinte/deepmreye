def volume_from_cifti(data, axis):
    import nibabel as nb
    import numpy as np 
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    data = data.T[axis.volume_mask]                          # Assume brainmodels axis is last, move it to front
    volmask = axis.volume_mask                               # Which indices on this axis are for voxels?
    vox_indices = tuple(axis.voxel[axis.volume_mask].T)      # ([x0, x1, ...], [y0, ...], [z0, ...])
    vol_data = np.zeros(axis.volume_shape + data.shape[1:],  # Volume + any extra dimensions
                        dtype=data.dtype)
    vol_data[vox_indices] = data                             # Fancy indexing"
    return nb.Nifti1Image(vol_data, axis.affine)             # Add affine for spatial interpretation


def surf_data_from_cifti(data, axis, surf_name):
    import nibabel as nb
    import numpy as np 
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            data = data.T[data_indices]                       # Assume brainmodels axis is last, move it to front
            vtx_indices = model.vertex                        # Generally 1-N, except medial wall vertices
            surf_data = np.zeros((vtx_indices.max() + 1,) + data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


def decompose_cifti(img):
    import numpy as np 
    data = img.get_fdata(dtype=np.float32)
    brain_models = img.header.get_axis(1)  
    return (volume_from_cifti(data, brain_models),
            surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_LEFT").T,
            surf_data_from_cifti(data, brain_models, "CIFTI_STRUCTURE_CORTEX_RIGHT").T)



#vol, left, right = decompose_cifti(test)
#print(vol.shape, left.shape, right.shape)