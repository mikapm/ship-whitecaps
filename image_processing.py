#!/usr/bin/env python3
"""
Some useful image processing functions.
"""

import os
import json
import cv2
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import skimage
from datetime import datetime as DT
from datetime import timedelta as TD
from scipy.interpolate import griddata
import motion_correction as swm
import gridding as swg
import plotting as swpl

def crop_horizon(im_stab, MC, K):
    """
    Crop horizon from stabilised image using
    self-parameters in swm.MotionCorrection object
    MC and intrinsic camera matrix K.
    Returns input image with horizon cropped out and 
    location of horizon in v pixels (rows).
    """
    # Get stabilised horizon line coordinates
    r_stab, theta_stab = swm.angles_to_horizon(inc=MC.incStab,
            roll=MC.rollStab, K=K)
    u_hor = np.array([0,im_stab.shape[1]-1])
    v_hor = swm.horizon_pixels(r=r_stab, theta=theta_stab, u=u_hor)
    v_hor = np.ceil(v_hor) # Round to integers
    # Crop out horizon from image
    im_crop = im_stab[int(v_hor[0]):,:]

    return im_crop, v_hor

def crop_borders(im_warped, crop_factor=1.0, debugging_fignm=None):
    """
    Returns warped image cropped to rectangle with
    maximum area excluding black borders due to warping.

    Assumes that no black borders are found at the
    top edge of the image (fine as long as horizon
    has been cropped out).

    Set crop_factor to < 1.0 to reduce the size of
    the cropped rectangle.

    Returns cropped image and cropping corners relative
    to input image.
    """
    # Binarize image (black=0)
    mask = im_warped > 0 
    # Coordinates of mask (i.e. non-black pixels)
    coords = np.argwhere(mask)
    # Find first row from the top with black border
    try:
        row0, _ = coords.min(axis=0)
    except ValueError:
        print('Error in swi.crop_borders(), plotting figure for'
                'debugging ... \n')
        print('coords: ', coords)
        print('mask: ', mask)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(im_warped)
        if debugging_fignm is not None:
            plt.savefig(debugging_fignm)
        else:
            plt.show()
        plt.close()
    # Find lowest row with non-black pixels
    row1 = np.squeeze(np.nonzero(np.sum(mask, axis=1)))[-1]
    # Loop through row0:row1 and calculate max rectangle
    # areas and coordinates for each row
    areas = [] # list to store areas
    corners = [] # List to store recangle corner coordinates
    for r in range(row0,row1):
        width = np.sum(mask[r,:]) # width of rectangle at row r
        # Find column coordinates of width
        nonzero_cols = np.argwhere(mask[r,:]!=0)
        col0 = int(nonzero_cols[0]) # lower left corner column
        col1 = int(nonzero_cols[-1]) # lower right corner column
        height = r # height of rectangle
        # Append lists with area and corners
        areas.append(width * height) 
        corners.append((r, col0, col1))
    # Find corners of rectangle with largest area
    # If several equal largest areas, take the first one.
    r_max, c_min, c_max = corners[int(np.atleast_1d(np.where(
        areas==np.max(areas))[0])[0])]
    r_min = 0 # Assume top row is free of black borders
    # Apply crop factor
    r_range = r_max - r_min # row range
    c_range = c_max - c_min # column range
    r_diff = r_range - np.floor(crop_factor*(r_range))
    c_diff = c_range - np.floor(crop_factor*(c_range))
    r_min, r_max = (int(r_min+r_diff), int(r_max-r_diff))
    c_min, c_max = (int(c_min+c_diff), int(c_max-c_diff))
    # Crop image
    im_cropped = im_warped[r_min:r_max, c_min:c_max]
    
    return im_cropped, (r_min, r_max, c_min, c_max)


def inscribed_rectangle(u_box_warped, v_box_warped):
    """
    Find inscribed regular rectangle inside warped rectangle.
    Note: does not maximize the area of the inscribed
    rectangle.

    Parameters:
        u_box_warped - 2D ndarray; u-coordinates (columns) 
                       of warped rectangle
        v_box_warped - 2D ndarray; v-coordinates (rows)
                       of warped rectangle
    Returns:
        rect_corners - corners of inscribed rectangle in
                       pixel coordinates as
                       (r_min, r_max, c_min, c_max)
    """
    # Get (row,col) corners of warped rectangle
    upper_left = (v_box_undist[:,1].min(),u_box_undist[:,1].max())
    upper_right = (v_box_undist[1,:].max(),u_box_undist[1,:].max())
    lower_right = (v_box_undist[:,-1].max(),u_box_undist[:,-1].max())
    lower_left = (v_box_undist[-1,:].min(),u_box_undist[-1,:].min())
    # Get rectangle corners (not necessarily max. area)
    r_min = np.max((int(upper_left[0]), int(upper_right[0])))
    r_max = np.min((int(lower_left[0]), int(lower_right[0])))
    #r_max = np.floor(np.min(lower_left[0], lower_right[0]))
    #c_min = np.floor(np.max(upper_left[1], lower_left[1]))
    c_min = np.max((int(upper_left[1]), int(lower_left[1])))
    #c_max = np.floor(np.min(upper_right[1], lower_right[1]))
    c_max = np.min((int(upper_right[1]), int(lower_right[1])))

    # TODO: Check if one of the sides can be extended to
    # make larger area

    return(r_min, r_max, c_min, c_max)
    

def img_to_grid(im, X, Y, dx, dy, x_grid=None, y_grid=None,
        return_mask=True, plot_grid=False, fname_grid=None,
        vertices=None, weights=None, **kwargs):
    """
    Regrid geo-rectified image to regular grid of resolution 
    (dx x dy). X and Y are the 2D irregular grids from which to
    interpolate.
    If return_mask is set to True a NaN mask will be returned in
    which NaNs are denoted by 0s and valid pixels by 255s (as
    required by cv2).
    Setting plot_grid to True makes plots of the x and y
    resolutions of the original X,Y grid.
    **kwargs as for scipy.interpolate.griddata(),
    e.g. method='linear'.
    """
    if x_grid is None:
        # Set limits
        xlim = (np.ceil(np.min(X)), np.floor(np.max(X)))
        ylim = (np.ceil(np.min(Y)), np.floor(np.max(Y)))
        # Make output x,y grid
        y_grid, x_grid = np.mgrid[ylim[0]:ylim[1]:dy, xlim[0]:xlim[1]:dx]
        y_grid = np.flip(y_grid) # "Flip" y_grid
    # Interpolate
    if vertices is not None:
        print('Interpolating image to regular grid with speed-up ... \n')
        ny, nx = x_grid.shape
        im_grid = swg.interpolate(im, vertices, weights)
        im_grid = im_grid.reshape(ny, nx)
    else:
        print('Interpolating image to regular grid ... \n')
        im_grid = griddata((X.flatten(),Y.flatten()), im.flatten(),
                (x_grid,y_grid), **kwargs)
    if plot_grid:
        print('Plotting original rectified image resolution ... \n')
        fig, ax = plt.subplots(ncols=2, figsize=(8,4))
        # X grid with 16 discrete color levels
        m0 = ax[0].pcolormesh(X, Y, np.diff(X, axis=1),
                cmap=plt.cm.get_cmap('RdGy', 16))
        fig.colorbar(m0, ax=ax[0], label='resolution [m]')
        ax[0].set_title('X resolution')
        ax[0].set_xlabel('m')
        ax[0].set_ylabel('m')
        # Y grid with 16 discrete color levels
        m1 = ax[1].pcolormesh(X, Y, -np.diff(Y, axis=0),
                cmap=plt.cm.get_cmap('RdGy', 16))
        fig.colorbar(m1, ax=ax[1], label='resolution [m]')
        ax[1].set_title('Y resolution')
        ax[1].set_xlabel('m')
        ax[1].set_ylabel('m')
        plt.tight_layout()
        if fname_grid is not None:
            plt.savefig(fname_grid)
        else:
            plt.show()
        plt.close()
    if return_mask:
        # Create mask for further use
        mask = ~np.isnan(im_grid)
        mask = mask.astype(np.uint8) # Zeros = masked pixels
        mask[mask==1]=255 # Unmasked pixels need a white mask
        return im_grid, x_grid, y_grid, mask
    else:
        return im_grid, x_grid, y_grid



def remove_gradients(im, sigma_thresh=3.25):
    """
    Remove background gradients from image by
    pre-thresholding following Brumer et al. (2017),
    DOI: 10.1175/JPO-D-17-0005.1, who state that
    'Removing background gradients was found to greatly
    improve subsequent whitecap detection via the typical
    automated brightness threshold techniques.'

    Parameters:
        im - ndarray; image(s) to threshold
        sigma_thresh - multiplier of pixel brightness standard 
                       deviation for pre- thresholding. Default 
                       as in Brumer et al. (2017).
    """
    # Copy input image for output
    im_nograd = im.copy()
    # Pre-threshold input image
    mean_brightness = np.mean(im)
    std_brightness = np.std(im)
    # Mask brightest pixels
    thresh = mean_brightness + sigma_thresh * std_brightness
    print(thresh)
    im_masked = np.ma.masked_array(im, mask=im>thresh)
    # Compute and subtract row and column means (ignoring
    # masked values)
#        for i in range(len(im_nograd[:,1])):
#            row_mean = im_masked[i,:].mean()
#            print(row_mean)
#            print('old: ', im_nograd[i,:])
#            im_nograd[i,:] -= int(row_mean)
#            print('new: ', im_nograd[i,:])
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0,
            tileGridSize=(50,50))
    im_nograd = clahe.apply(im_nograd)

    return im_nograd


def remove_background_gradients(images, mask=None, cv_mask=True,
        plot=False, xgrid=None, ygrid=None, figname=None):
    """
    Remove gradients in background pixel intensity by pixel-wise
    division by pixel-wise median.
    """
    print('Removing image background gradients ... \n')
    print('images shape: ', images.shape)

    # Make mask of same shape as images
    mask3d = np.broadcast_to(mask, images.shape)
    # Mask images
    if mask is not None:
        if cv_mask:
            # In OpenCV mask 0 means masked value
            images = np.ma.masked_array(images, mask=mask3d==0)
            mask2d = mask==0
            print('MASK')
        else:
            # Regular mask where 1 means masked
            images = np.ma.masked_array(images, mask=mask3d==1)
            mask2d = mask
    # Calculate row-wise mean intensity and divide im_grid by it to remove
    # pixel intensity gradients
    print('Computing row-wise mean ... \n')
    #pixel_median = np.nanmedian(images, axis=0)
    row_mean = np.nanmean(images, axis=(0,2))
    # Make row mean grid
    row_mean = np.tile(np.vstack(row_mean), images.shape[2])
    # Mask row mean 
    row_mean = np.ma.masked_array(row_mean, mask=mask2d)
    print(row_mean)
    print(np.nanmax(row_mean))
    print(np.nanmin(row_mean))
    print(np.nanmax(images))
    print(np.nanmin(images))
    # Divide by row mean
    im_divmed = images.copy() / row_mean
    # Rescale intensities to min, max intensity range of original image grids
    in_range = (np.nanmin(im_divmed), np.nanmax(im_divmed))
    #in_range = (np.nanmin(images), np.nanmax(images))
    #in_range = (0, 255)
    out_range = (np.nanmin(images), np.nanmax(images))
    #out_range = (0, 255)
    print('Rescaling pixel intensities ... \n')
#    im_divmed[im_divmed < 0] = 0
#    im_divmed[im_divmed > 1] = 1
#    im_divmed *= 255
#    im_out = im_divmed
    
    im_out = skimage.exposure.rescale_intensity(im_divmed, in_range,
            out_range).astype(np.uint8)

    # Plot if requested
    if plot:
        fig, ax = plt.subplots(ncols=3, figsize=(18,6))
        if xgrid is None:
            p0 = ax[0].imshow(row_mean, cmap='gray')
            p1 = ax[1].imshow(im_out[0], cmap='gray')
            p2 = ax[2].imshow(images[0], cmap='gray')
        else:
            vmin=np.nanmin(images[0])
            vmax=np.nanmax(images[0])
            p0 = ax[0].pcolormesh(xgrid, ygrid, row_mean, cmap='gray',)
            p1 = ax[1].pcolormesh(xgrid, ygrid, images[0], cmap='gray',
                    vmin=vmin, vmax=vmax)
            p2 = ax[2].pcolormesh(xgrid, ygrid,
                    np.ma.masked_array(im_out[0], mask=mask2d),
                    cmap='gray', vmin=vmin, vmax=vmax)
                    #vmax=255)
        ax[0].set_title('gradient')
        ax[1].set_title('orig')
        ax[2].set_title('edited')
        fig.colorbar(p0, ax=ax[0], label='row mean px intensity')
        fig.colorbar(p1, ax=ax[1], label='px intensity')
        fig.colorbar(p2, ax=ax[2], label='px intensity')
        if figname is not None:
            plt.savefig(figname)
        else:
            plt.show()
        plt.close()

    return im_out


def threshold_KM11(DS, method='global', smooth=False,
        bins=range(0,256), d=7, sigmaColor=75, sigmaSpace=75,
        plot=False, n_plot=50, figname=None, hist=None,
        x_grid=None, y_grid=None, mask=None, correct_background=False,
        outdir=None, time_slice=None, x_slice=None, y_slice=None):
    """
    Find image threshold for whitecap detection based on
    histogram method of Kleiss and Melville (2011; KM11):
    "The Analysis of Sea Surface Imagery for Whitecap Kinematics"

    Parameters:
        DS - xarray Dataset; dataset with image grids and mask
        method - str; one of ['global', 'otsu', ]
        smooth - bool; if True, smooth image with bilateral
                 filter before thresholding
        bins - range; histogram bin range
        (d,sigmaColor,sigmaSpace) - as for cv2.bilateralFilter()
        plot - bool; Set to true to make plots of thresholding
               details
               n_plot - int; number of frames to plot
        figname - str; figure name for saving plot in outdir
        hist - (256x1) ndarray; pre-computed histogram. If not
               given a histogram will be computed from the
               input image(s).
        x_grid - ndarray; grid of x coordinate of image grids. If None,
                 will use DS.x_grid
        y_grid - ndarray; grid of y coordinate of image grids. If None,
                 will use DS.x_grid
        mask - ndarray; mask for boundaries of non-rectangular grid. If
               None, will use DS.mask
        correct_background - bool; if True, divide each image grid by
                             pixel-wise (time) median intensity and 
                             standardize intensities using
                             skimage.exposure.rescale_intensity().
        x_slice - slice; x range to analyse
        y_slice - slice; y range to analyse
    Returns:
        wc_cov_20 - whitecap fraction based on threshold.
    """
    # Define mask and x,y grids
    if x_grid is None:
        x_grid = DS.xgrid.sel(x=x_slice, y=y_slice).values
    if y_grid is None:
        y_grid = DS.ygrid.sel(x=x_slice, y=y_slice).values
    if mask is None:
        mask = DS.mask.sel(x=x_slice, y=y_slice).values

    # Load image grids to memory
    if time_slice is not None:
        images = DS.im_grid.sel(time=time_slice, x=x_slice, y=y_slice).where(
                mask != 0) 
    else:
        images = DS.im_grid.sel(x=x_slice, y=y_slice).where(mask != 0) 
    if correct_background:
        images = swi.remove_background_gradients(images, mask=mask, cv_mask=True)

    # Compute composite histogram 
    if hist is None:
        print('Computing histogram ... \n')
        hist = np.zeros(256).reshape(-1,1)
        for i,grid in enumerate(images):
            if smooth:
                # Smooth with bilateral filter 
                grid = cv2.bilateralFilter(grid, d, sigmaColor,
                        sigmaSpace)
            # cv2 does not like NaNs -> nan_to_num(grid) converts NaNs to
            # zeros.
            hist += cv2.calcHist([np.nan_to_num(grid).astype(np.uint8)], [0],
                    mask, [len(bins)], [bins[0], bins[-1]+1])
#        if outdir is not None:
#            # Save histogram
#            if correct_background:
#                fn_hist = os.path.join(outdir, 'hist_bgcorr.txt') # filename
#            else:
#                fn_hist = os.path.join(outdir, 'hist.txt') # filename
#            if not os.path.isfile(fn_hist):
#                print('Saving histogram ... \n')
#                np.savetxt(fn_hist, hist)

    # Normalize histogram
    hist = hist.ravel()/(hist * bins).sum()
    Q = hist.cumsum()
    # Complementary cumulative distribution (K&M eq. 8)
    W = 1 - Q
    # Logarithm of W
    L = np.log10(W)
    # Smooth L
    L_s = sp.signal.savgol_filter(L, 11, 2)
    # Take first derivative and smooth again
    L_p = sp.signal.savgol_filter(np.gradient(L_s), 11, 2)
    # Take second derivative and smooth
    L_pp = sp.signal.savgol_filter(np.gradient(L_p), 11, 2)
    # Find peaks of L_pp
    peak_ind = sp.signal.find_peaks(L_pp,
            prominence=np.std(L_pp))[0]
    peak_ind = peak_ind[-1] # want last peak
    # Find index of end of region of positive curvature, i.e. 
    # point that first falls below 20% of peak value (following
    # Kleiss & Melville, 2011)
    peak_value = L_pp[peak_ind] # Value of L_pp peak
    # Make dictionary of thresholds from 1-50%
    thresh_dict = {}
    for th in np.linspace(0.01 ,0.5, 50):
        thresh_i = np.atleast_1d(peak_ind + np.squeeze(np.where(
            L_pp[peak_ind:]<th*peak_value))[0])
        thresh_dict['thresh_{}'.format(int(th*100))] = int(thresh_i)
#    if outdir is not None:
#        # Save dict as .json
#        print('Saving thresholds to .json ... \n')
#        fname_json = os.path.join(outdir, 'thresholds_KM11.json')
#        with open(fname_json, 'w') as fp:
#            json.dump(thresh_dict, fp)

    if plot:
        plotdir = os.path.join(outdir, 'Thresholded_images')
        print('Plotdir: ', plotdir)
        if not os.path.isdir(plotdir):
            os.mkdir(plotdir)
        swpl.plot_whitecap_thresholds(DS, hist, L_pp, thresh_dict, mask, 
                thresholds=[1,5,10], outdir=plotdir, figname=figname, 
                time_slice=time_slice, x_slice=x_slice, y_slice=y_slice)

    return thresh_dict, hist

def compute_W_tot(DS, thresh_dict, thresh_str, mask_cv, plot=False,
        outdir=None, time_slice=None, x_slice=None, y_slice=None):
    """
    Calculates total (active + stage B) whitecap coverage 
    fraction on input images based on the brightness intensity
    threshold thresh (an integer between 0 and 255).
    Save timeseries of frame-wise W in netcdf file.

    Parameters:
        thresh - int; brightness threshold (0-255)
        DS - xarray dataset with image grids 
        mask_cv - ndarray; OpenCV mask, where 0 means masked, 255 means
                  unmasked
        plot - bool; set to True to generate plot of W as timeseries
    """
    thresh = thresh_dict[thresh_str]
    # Mask image grids according to mask
    print('Masking image grids ... \n')
    if time_slice is not None:
        images = DS.im_grid.sel(time=time_slice, x=x_slice, y=y_slice).where(
                mask_cv != 0)
    else:
        images = DS.im_grid.sel(x=x_slice, y=y_slice).where(mask_cv != 0)
    # Average brightness of all grids
#        im_mean = np.floor(images.mean(skipna=True).values)
    # Make xr.DataArray for W
    da = xr.DataArray(np.zeros(len(images)), 
            coords=[DS.time.sel(time=time_slice).values], dims=['time'])
    # Convert DataArray to dataset for later conversion to netcdf
    ds = da.to_dataset(name='w_tot')
    # Loop over image grids and compute whitecap coverage for each
    # grid.
    print('Thresholding image grids and computing W_tot ... \n')
    for i,im in enumerate(images.values):
        # Total whitecap coverage is computed as the number of 
        # grid cells whose brightness exceeds the given
        # threshold divided by the total number of non-Nan pixels in
        # the grid.
        ds.w_tot[i] = ((im > thresh).sum() / 
                np.count_nonzero(~np.isnan(im)))
        
    # Save ds to netcdf
    if outdir is not None:
#        nc_name = os.path.join(outdir, 'w_tot_{0}.nc'.format(thresh_str))
        png_name = os.path.join(outdir, 'w_tot_{0}.png'.format(thresh_str))
#    ds.to_netcdf(nc_name)

    # Plot if requested
    if plot:
        # Read netcdf file
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        #ax.plot(pd.to_datetime(DS.time.values), W)
        ds.w_tot.plot(ax=ax)
        plt.tight_layout()
        plt.savefig(png_name)
        plt.close()

    return ds




