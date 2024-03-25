import numpy as np
def get_roi_verts_hemi(fn,subject,rois):
    """
    load an surface image, and return vertex from ROIs only from the corresponding 
    hemisphere

    Parameters
    ----------
    fn : surface filename
    subject : subject 
    rois : list of rois you want extract 
    
    Returns
    -------
    img : the image load from fn   
    data_roi : numpy rois data 
              2 dim (time x vertices from all the rois)  
              
    roi_idx : indices of the rois vertices 
    
    
    data_hemi : numpy stacked data
                2 dim (time x vertices)    
    """
    
    import cortex
    from surface_utils import load_surface

    
    
    # import data 
    img, data = load_surface(fn=fn)
    len_data = data.shape[1]
    
    # export masks 
    roi_verts = cortex.get_roi_verts(subject=subject, 
                                     roi= rois, 
                                     mask=True
                                    )
    na_vertices = np.isnan(data).any(axis=0)
    
    # create a brain mask  
    brain_mask = np.any(list(roi_verts.values()), axis=0)
    
    # create a hemi mask  
    if 'hemi-L' in fn:
        hemi_mask = brain_mask[:len_data]
        for i, na_vertices in enumerate(na_vertices):
            hemi_mask[i] = not na_vertices and hemi_mask[i]
        
    elif 'hemi-R' in fn: 
        hemi_mask = brain_mask[-len_data:]
        for i, na_vertices in enumerate(na_vertices):
            hemi_mask[i] = not na_vertices and hemi_mask[i]
    else: 
        hemi_mask = brain_mask
        
    roi_idx = np.where(hemi_mask)[0]
    
    data_roi = data[:,hemi_mask]

        
    return img, data, data_roi, roi_idx

def get_roi_masks_hemi(fn,subject,rois):
    """
    Acces to a single hemisphere rois masks 

    Parameters
    ----------
    fn : surface filename
    subject : subject 
    rois : list of rois you want extract 
    
    Returns
    -------
    rois_masks : A dictionary where the keys represent the ROIs 
    and the values correspond to the respective masks for each hemisphere.
             
    hemi : The correponding hemisphere. 
  
    """
    import cortex
    from surface_utils import load_surface

    # import data 
    img, data = load_surface(fn=fn)
    len_data = data.shape[1]  
    
    # export masks 
    roi_verts = cortex.get_roi_verts(subject=subject, 
                                     roi= rois, 
                                     mask=True
                                    )
    # create a hemi mask  
    if 'hemi-L' in fn:
        hemi = 'hemi-L'
        rois_masks = {roi: data[:len_data] for roi, data in roi_verts.items()}
        
    elif 'hemi-R' in fn:
        hemi = 'hemi-R'
        rois_masks = {roi: data[-len_data:] for roi, data in roi_verts.items()}
          
    return rois_masks, hemi

def load_surface_pycortex(L_fn=None, R_fn=None, brain_fn=None, return_img=None, return_hemi_len=None):
    """
    Load a surface image independently if it's CIFTI or GIFTI, and return 
    concatenated data from the left and right cortex

    Parameters
    ----------
    L_fn : gifti left hemisphere filename
    R_fn : gifti right hemisphere filename
    brain_fn : brain data in cifti format
    return_img : whether to include img in the return
    return_hemi_len : whether to include hemisphere lengths in the return
    
    Returns
    -------
    result : numpy array or list
        data_concat : numpy stacked data of the two hemisphere. 
                      2 dim (time x vertices)
        (optional) img_L : surface image data for the left hemisphere
        (optional) img_R : surface image data for the right hemisphere
        (optional) len_L : length of the left hemisphere data
        (optional) len_R : length of the right hemisphere data
    """
    
    from surface_utils import load_surface
    from cifti_utils import decompose_cifti
    
    if L_fn and R_fn: 
        img_L, data_L = load_surface(L_fn)
        len_L = np.shape(data_L)[1]
        img_R, data_R = load_surface(R_fn)
        len_R = np.shape(data_R)[1]
        data_concat = np.concatenate((data_L, data_R), axis=1)
        
    elif brain_fn:
        img, mat = load_surface(brain_fn)
        vol, data_L, data_R = decompose_cifti(img)
        data_concat = np.concatenate((data_L, data_R), axis=1)

    if return_img is None and return_hemi_len is None:
        return data_concat

    result = [data_concat]

    if return_img:
        result.append(img_L)
        result.append(img_R)

    if return_hemi_len:
        result.append(len_L)
        result.append(len_R)

    return result
    

def set_pycortex_config_file(cortex_folder):

    # Import necessary modules
    import os
    import cortex
    from pathlib import Path

    # Define the new database and colormaps folder
    pycortex_db_folder = "{}/db/".format(cortex_folder)
    pycortex_cm_folder = "{}/colormaps/".format(cortex_folder)
    
    # Get pycortex config file location
    pycortex_config_file  = cortex.options.usercfg

    # Create name of new config file that will be written
    new_pycortex_config_file = pycortex_config_file[:-4] + '_new.cfg'

    # Create the new config file
    Path(new_pycortex_config_file).touch()

    # Open the config file in read mode and the newly created one in write mode.
    # Loop over every line in the original file and copy it into the new one.
    # For the lines containing either 'filestore' or 'colormap', it will
    # change the saved folder path to the newly created one above (e.g. pycortex_db_folder)
    with open(pycortex_config_file, 'r') as fileIn:
        with open(new_pycortex_config_file, 'w') as fileOut:

            for line in fileIn:

                if 'filestore' in line:
                    newline = 'filestore=' + pycortex_db_folder
                    fileOut.write(newline)
                    newline = '\n'

                elif 'colormaps' in line:
                    newline = 'colormaps=' + pycortex_cm_folder
                    fileOut.write(newline)
                    newline = '\n'

                else:
                    newline = line

                fileOut.write(newline)

    
    # Renames the original config file als '_old' and the newly created one to the original name
    os.rename(pycortex_config_file, pycortex_config_file[:-4] + '_old.cfg')
    os.rename(new_pycortex_config_file, pycortex_config_file)
    return None

def draw_cortex(subject,xfmname,data,vmin,vmax,description,cortex_type='VolumeRGB',cmap='Viridis',cbar = 'discrete',cmap_steps = 255,\
                alpha = None,depth = 1,thick = 1,height = 1024,sampler = 'nearest',\
                with_curvature = True,with_labels = False,with_colorbar = False,\
                with_borders = False,curv_brightness = 0.95,curv_contrast = 0.05,add_roi = False,\
                roi_name = 'empty',col_offset = 0, zoom_roi = None, zoom_hem = None, zoom_margin = 0.0,cbar_label=''):
    """
    Plot brain data onto a previously saved flatmap.
    Parameters
    ----------
    subject             : subject id (e.g. 'sub-001')
    xfmname             : xfm transform
    data                : the data you would like to plot on a flatmap
    cmap                : colormap that shoudl be used for plotting
    vmins               : minimal values of 1D 2D colormap [0] = 1D, [1] = 2D
    vmaxs               : minimal values of 1D/2D colormap [0] = 1D, [1] = 2D
    description         : plot title
    cortex_type         : cortex function to create the volume (VolumeRGB, Volume2D, VertexRGB)
    cbar                : color bar layout
    cbar_label          : colorbar label
    cmap_steps          : number of colormap bins
    alpha               : alpha map
    depth               : Value between 0 and 1 for how deep to sample the surface for the flatmap (0 = gray/white matter boundary, 1 = pial surface)
    thick               : Number of layers through the cortical sheet to sample. Only applies for pixelwise = True
    height              : Height of the image to render. Automatically scales the width for the aspect of the subject's flatmap
    sampler             : Name of sampling function used to sample underlying volume data. Options include 'trilinear', 'nearest', 'lanczos'
    with_curvature      : Display the rois, labels, colorbar, annotated flatmap borders, or cross-hatch dropout?
    with_labels         : Display labels?
    with_colorbar       : Display pycortex colorbar?
    with_borders        : Display borders?
    curv_brightness     : Mean brightness of background. 0 = black, 1 = white, intermediate values are corresponding grayscale values.
    curv_contrast       : Contrast of curvature. 1 = maximal contrast (black/white), 0 = no contrast (solid color for curvature equal to curvature_brightness).
    add_roi             : add roi -image- to overlay.svg
    roi_name            : roi name
    col_offset          : colormap offset between 0 and 1
    zoom_roi            : name of the roi on which to zoom on
    zoom_hem            : hemifield fo the roi zoom
    zoom_margin         : margin in mm around the zoom
    Returns
    -------
    braindata - pycortex volumr or vertex file
    """
    
    import cortex
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib import cm
    import matplotlib as mpl
    import ipdb
    
    deb = ipdb.set_trace
    
    # define colormap
    try: base = plt.cm.get_cmap(cmap)
    except: base = cortex.utils.get_cmap(cmap)
    
    if '_alpha' in cmap: base.colors = base.colors[1,:,:]
    val = np.linspace(0, 1,cmap_steps+1,endpoint=False)
    colmap = colors.LinearSegmentedColormap.from_list('my_colmap',base(val), N = cmap_steps)
    
    if cortex_type=='VolumeRGB':
        # convert data to RGB
        vrange = float(vmax) - float(vmin)
        norm_data = ((data-float(vmin))/vrange)*cmap_steps
        mat = colmap(norm_data.astype(int))*255.0
        alpha = alpha*255.0

        # define volume RGB
        braindata = cortex.VolumeRGB(channel1 = mat[...,0].T.astype(np.uint8),
                                     channel2 = mat[...,1].T.astype(np.uint8),
                                     channel3 = mat[...,2].T.astype(np.uint8),
                                     alpha = alpha.T.astype(np.uint8),
                                     subject = subject,
                                     xfmname = xfmname)
    elif cortex_type=='Volume2D':
        braindata = cortex.Volume2D(dim1 = data.T,
                                 dim2 = alpha.T,
                                 subject = subject,
                                 xfmname = xfmname,
                                 description = description,
                                 cmap = cmap,
                                 vmin = vmin[0],
                                 vmax = vmax[0],
                                 vmin2 = vmin[1],
                                 vmax2 = vmax[1])
    elif cortex_type=='VertexRGB':
        
        # convert data to RGB
        vrange = float(vmax) - float(vmin)
        norm_data = ((data-float(vmin))/vrange)*cmap_steps
        mat = colmap(norm_data.astype(int))*255.0
        alpha = alpha*255.0
        
        # define Vertex RGB
        braindata = cortex.VertexRGB( red = mat[...,0].astype(np.uint8),
                                      green = mat[...,1].astype(np.uint8),
                                      blue = mat[...,2].astype(np.uint8),
                                      subject = subject,
                                      alpha = alpha.astype(np.uint8))
    elif cortex_type=='Vertex':
        
        # define Vertex 
        braindata = cortex.Vertex(data = data,
                                 subject = subject,
                                 description = description,
                                 cmap = cmap,
                                 vmin = vmin,
                                 vmax = vmax)
        
    braindata_fig = cortex.quickshow(braindata = braindata,
                                     depth = depth,
                                     thick = thick,
                                     height = height,
                                     sampler = sampler,
                                     with_curvature = with_curvature,
                                     with_labels = with_labels,
                                     with_colorbar = with_colorbar,
                                     with_borders = with_borders,
                                     curvature_brightness = curv_brightness,
                                     curvature_contrast = curv_contrast)
    if cbar == 'polar':
        try: base = plt.cm.get_cmap(cmap)
        except: base = cortex.utils.get_cmap(cmap)
        val = np.arange(1,cmap_steps+1)/cmap_steps - (1/(cmap_steps*2))
        val = np.fmod(val+col_offset,1)
        colmap = colors.LinearSegmentedColormap.from_list('my_colmap', base(val), N=cmap_steps)
        cbar_axis = braindata_fig.add_axes([0.5, 0.07, 0.8, 0.2], projection='polar')
        norm = colors.Normalize(0, 2*np.pi)
        t = np.linspace(0,2*np.pi,200,endpoint=True)
        r = [0,1]
        rg, tg = np.meshgrid(r,t)
        im = cbar_axis.pcolormesh(t, r, tg.T,norm=norm, cmap=colmap)
        cbar_axis.set_yticklabels([])
        cbar_axis.set_xticklabels([])
        cbar_axis.set_theta_zero_location("W")
        cbar_axis.spines['polar'].set_visible(False)

    elif cbar == 'ecc':
        colorbar_location = [0.5, 0.07, 0.8, 0.2]
        n = 200
        cbar_axis = braindata_fig.add_axes(colorbar_location, projection='polar')
        t = np.linspace(0,2*np.pi, n)
        r = np.linspace(0,1, n)
        rg, tg = np.meshgrid(r,t)
        c = tg
        im = cbar_axis.pcolormesh(t, r, c, norm = mpl.colors.Normalize(0, 2*np.pi), cmap=colmap)
        cbar_axis.tick_params(pad=1,labelsize=15)
        cbar_axis.spines['polar'].set_visible(False)
        box = cbar_axis.get_position()
        cbar_axis.set_yticklabels([])
        cbar_axis.set_xticklabels([])
        axl = braindata_fig.add_axes([0.97*box.xmin,0.5*(box.ymin+box.ymax), box.width/600,box.height*0.5])
        axl.spines['top'].set_visible(False)
        axl.spines['right'].set_visible(False)
        axl.spines['bottom'].set_visible(False)
        axl.yaxis.set_ticks_position('right')
        axl.xaxis.set_ticks_position('none')
        axl.set_xticklabels([])
        axl.set_yticklabels(np.linspace(vmin, vmax, 3),size = 'x-large')
        axl.set_ylabel('$dva$\t\t', rotation=0, size='x-large')
        axl.yaxis.set_label_coords(box.xmax+30,0.4)
        axl.patch.set_alpha(0.5)

    elif cbar == 'discrete':
        colorbar_location= [0.8, 0.05, 0.1, 0.05]
        cmaplist = [colmap(i) for i in range(colmap.N)]
        bounds = np.linspace(vmin, vmax, cmap_steps + 1)  
        bounds_label = np.linspace(vmin, vmax, 3)
        norm = mpl.colors.BoundaryNorm(bounds, colmap.N)

        cbar_axis = braindata_fig.add_axes(colorbar_location)
        cb = mpl.colorbar.ColorbarBase(cbar_axis, cmap=colmap, norm=norm, ticks=bounds_label, boundaries=bounds,orientation='horizontal')
        cb.set_label(cbar_label,size='x-large')

    elif cbar == '2D':
        cbar_axis = braindata_fig.add_axes([0.8, 0.05, 0.15, 0.15])
        base = cortex.utils.get_cmap(cmap)
        cbar_axis.imshow(np.dstack((base.colors[...,0], base.colors[...,1], base.colors[...,2],base.colors[...,3])))
        cbar_axis.set_xticks(np.linspace(0,255,3))
        cbar_axis.set_yticks(np.linspace(0,255,3))
        cbar_axis.set_xticklabels(np.linspace(vmin[0],vmax[0],3))
        cbar_axis.set_yticklabels(np.linspace(vmax[1],vmin[1],3))
        cbar_axis.set_xlabel(cbar_label[0], size='x-large')
        cbar_axis.set_ylabel(cbar_label[1], size='x-large')

    
    # add to overlay
    if add_roi == True:
        cortex.utils.add_roi(   data = braindata,
                                name = roi_name,
                                open_inkscape = False,
                                add_path = False,
                                depth = depth,
                                thick = thick,
                                sampler = sampler,
                                with_curvature = with_curvature,
                                with_colorbar = with_colorbar,
                                with_borders = with_borders,
                                curvature_brightness = curv_brightness,
                                curvature_contrast = curv_contrast)

    return braindata