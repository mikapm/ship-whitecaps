#!/usr/bin/env python3
"""
Calculate whitecap fraction from shipboard
images. Perform motion correction based on
horizon detection.
TODO: use IMU for motion correction.
"""

import sys
import os
import glob
import cv2
import json
import numpy as np
import scipy as sp
import scipy.io as spio
from scipy import ndimage as ndi
import skimage
import pandas as pd
import xarray as xr
import matplotlib as mpl
mpl.use('Agg') # Avoid displaying images
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import motion_correction as swm
import image_processing as swi
import gridding as swg
import netCDF4 as nc
from datetime import datetime as DT
from datetime import timedelta as TD

# Edit the font, font size, and axes width
mpl.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2

def replace_1d(ts, spike_mask, replace_single='linear',
        replace_multi='linear'):
    """
    Replace 1D data by specified interpolation method. 
    Designed to be used together with sw_pyfuns.despike.

    Parameters:
        ts - time series to replace and interpolate
        spike_mask - bool array; mask of points to replace,
                     0 for bad points
        replace_single - str; method of interpolation
                         for single-point spikes
        replace_multi - str; method of interpolation
                        for multi-point spikes

    Returns:
        ts_out - interpolated time series
    """

    # Initialise output array
    ts_out = ts.copy()

    # First interpolate all masked values linearly
    t = np.arange(len(ts)) # Time (x) axis
    ts_out = np.interp(t, t[spike_mask], ts_out[spike_mask])
    
    return ts_out


def tukey_53h(ts, k=1.0, replace_single='linear',
        replace_multi='last_nonsp', otsu_k=False):
    """
    Algorithm also described in Goring and Nikora (2002) and
    based on Otnes and Enochson (1978). Uses the principle of
    the robustness of the median as an estimator of the mean.
    Name in G&N02: Tukey 53H.

    This function is modified to check for multiple-point
    constant spikes ("plateaus") if two detected spike indices
    are consecutive. This modification is due to the ubiquity 
    of such spikes in the LASAR array data, and the inability
    of the original Tukey 53H algorithm to detect such spikes
    (see e.g. the figure tukey_53_illustration.pdf in 
    $PHDDIR/latex/texts/lasar_preprocessing/despiking_examples).

    Parameters: 
        ts - the time series (1D) array to despike
        k - scaling parameter for sigma, usually set to around 1.5
            (https://cotede.readthedocs.io/en/latest/qctests.html)
        replace_single - str; interpolation method for single-point spikes
        replace_multi - str; interpolation method for multi-point spikes 
        otsu_k - bool; use Otsu's method to find optimal threshold k
                 NB: Doesn't give good results for .wa4 LASAR data

    Returns:
        u - despiked time series       
        spike_mask - mask for detected spikes
    """
    # Copy the input time series
    u = ts.copy() 
    u = np.array(u)
    # Calculate sample variance
    sigma2 = np.var(u)
    # Standard deviation
    sigma = np.sqrt(sigma2)

    # 1. Compute the median u1 of the five points from i-2 to i+2
    u1 = np.zeros_like(u)
    for i, sample in enumerate(u[2:-2]):
        u1[i+2] = np.median(u[i:i+5])
    # End points with mirroring boundary conditions
    u1[0] = np.median(np.concatenate((u[:3],u[1:3]), axis=None))
    u1[1] = np.median(np.concatenate((u[:4],u[1]), axis=None))
    u1[-1] = np.median(np.concatenate((u[-3:],u[-3:-1]), axis=None))
    u1[-2] = np.median(np.concatenate((u[-4:],u[-2]), axis=None))

    # 2. Compute u2: the 3-point median of i-1 to i+1 of u1
    u2 = np.zeros_like(u)
    for i, sample in enumerate(u1[1:-1]):
        u2[i+1] = np.median(u1[i:i+3])
    # End points with mirroring boundary conditions
    u2[0] = np.median([u1[0], u[1], u[1]])
    u2[-1] = np.median([u[-1], u[-2], u[-2]])

    # 3. Compute a Hanning smoothing filter u3
    u3 = np.zeros_like(u)
    for i, sample in enumerate(u2[1:-1]):
        u3[i+1] = 0.25*(u2[i] + 2*sample + u2[i+2])
    # End points with mirroring boundary conditions
    u3[0] = 0.25*(u2[1] + 2*u2[0] + u2[1])
    u3[-1] = 0.25*(u2[-2] + 2*u2[-1] + u2[-2])

    # If specified, use Otsu's method to find the optimal threshold
    # NB! Doesn't give good results for LASAR data!
    if otsu_k:
        k = otsu_1979_threshold(np.abs(u-u3))
        # Don't use std if using Otsu-threshold
        sigma = 1

    # 4. Find spikes by taking the abs. difference between the
    #    original and the filtered data and comparing
    #    that to k*sigma
    # Make 1D array for masking detected spikes
    spike_mask = np.ones(len(ts), dtype=bool)
    for i, sample in enumerate(u):
        if np.abs(sample-u3[i]) > k*sigma:
            spike_mask[i] = 0

    # Replace spikes by linear interpolation
    # TODO include other interpolation methods
    if (spike_mask != 1).any():
        # Replace spikes
        u = replace_1d(u, spike_mask, 
                replace_single=replace_single,
                replace_multi=replace_multi)

    return u, spike_mask


def otsu_1979_threshold(diff_signal):
    """
    Algorithm developed by Otsu (1979): A Threshold
    Selection Method from Gray-Level Histograms, for
    optimal threshold selection.

    This code is based on the adaptation of Otsu's 
    method by Feuerstein et al. (2009): Practical
    Methods for Noise Removal ... in a Matlab code
    distributed as supporting information to that
    paper here:
    https://pubs.acs.org/doi/suppl/10.1021/ac900161x/
    suppl_file/ac900161x_si_001.pdf

    Otsu's method is based on a histogram of a 
    difference signal, and computes the threshold
    that best separates two data classes: spikes
    and the remaining signal. The optimal threshold is 
    that which minimises the within-class variance or,
    equivalently, maximises the between-class variance
    (Feuerstein et al., 2009).

    ------------------------------------------------------
    ------------------------------------------------------

    From Otsu (1979):
    The fundamental principle is to divide a signal
    (or, originally, image pixels) into two classes:
    background (C0) and objects (C1), or e.g. spikes and
    remaining signal. The classes are separated by a threshold
    level k such that C0 includes elements with pixel 
    levels (or signal amplitudes) [1, ..., k] and C1 has
    the elements [k+1, ..., L], where L is the total
    number of pixels. This gives the probabilities of
    class occurrence

    pr_C0 = \sum_{i=1}^{k} p_i
    pr_C1 = \sum_{i=k+1}^{L} p_i = 1 - pr_C0

    where p_i = n_i/N is the normalised histogram, or
    pdf, of the pixel levels.

    Following Feuerstein et al. (2009), the initial
    threshold is set to the level at k=2.

    After computing the class probabilities, the class
    means are computed as

    m_C0 = \sum_{i=1}^{k} i*p_i / pr_C0
    m_C1 = \sum_{i=k+1}^{L} i*p_i / pr_C1
    
    Now, the between-class variance can be computed as

    var_C0C1 = pr_C0*pr_C1 * (m_C0 - m_C1)**2

    The above three steps are repeated for every threshold
    index k. The optimal threshold opt_thresh is the one that
    maximises var_C0C1.

    ------------------------------------------------------
    ------------------------------------------------------

    Parameters:
        diff_signal: difference b/w raw and filtered
                     signal
        
    Returns:
        opt_thresh - optimal threshold for diff_signal
    """

    L = len(diff_signal)

    # Make a histogram of the difference signal
    hist, bin_edges = np.histogram(diff_signal)

    # Normalise hist by length of difference signal
    hist = hist/L

    # Use bin centres instead of bin edges (as in Matlab)
    bin_cents = bin_edges[:-1] + np.diff(bin_edges)/2
    N = len(bin_cents)

    # Initialise arrays
    thresholds = np.zeros(N)
    pr_C0 = np.zeros(N)
    pr_C1 = np.zeros(N)
    m_C0 = np.zeros(N)
    m_C1 = np.zeros(N)
    var_C0C1 = np.zeros(N)

    # Initial threshold: k=2
    thresholds[0] = bin_cents[1]
    # Initial probabilities
    pr_C0[0] = np.sum(hist[0])
    pr_C1[0] = 1 - pr_C0[0]
    # Initial means
    m_C0[0] = np.sum(bin_cents[0] * hist[0] / pr_C0[0])
    m_C1[0] = np.sum(bin_cents[1:] * hist[1:] / pr_C1[0])

    # Compute remaining probabilities of class occurrence
    # using remaining thresholds (bin centres), and test
    # the effect of different thresholds on between-class
    # variance.
    for i in range(1,N-1):
        thresholds[i] = bin_cents[i+1]
        pr_C0[i] = np.sum(hist[:i+1])
        pr_C1[i] = 1 - pr_C0[i]
        m_C0[i] = np.sum(bin_cents[:i+1]*hist[:i+1]/pr_C0[i])
        m_C1[i] = np.sum(bin_cents[i+1:]*hist[i+1:]/pr_C1[i])
        var_C0C1[i] = pr_C0[i]*pr_C1[i]*(m_C0[i]-m_C1[i])**2

    # Find the maximum between-class variance and its index k
    var_C0C1_max = np.max(var_C0C1)
    k = np.where(var_C0C1 == var_C0C1_max)

    # The optimal threshold maximises the between-class variance
    # If multiple equal max. variances, use the first one (as by
    # default in Matlab(?)).
    opt_thresh = thresholds[k][0]

    return opt_thresh


class WhiteCapFraction():
    """ Main class. """
    def __init__(self,
            datadir='/home/mikapm/stereo-wave/WhiteCapFraction',
            imgtype='pgm', 
            im_grid_res=80, # Image grid resolution in cm (!)
            imgdir=None, 
            griddir=None, # Image grid directory
            outdir=None,
            gridfile=None, # image grid netCDF file path
            histfile=None, # combined image histogram .txt file
            skip_imgdir=False, # Set to True for testing w/o input images
            ):
        self.datadir = datadir
        self.imgtype = imgtype # Image file extension
        # Input directory
        if imgdir is None:
            self.imgdir = os.path.join(datadir, 'Images')
        else:
            self.imgdir = imgdir
        if not skip_imgdir:
            if not os.path.isdir(self.imgdir):
                raise ValueError('Input directory not found.')
            # List input images
            self.images = sorted(glob.glob(os.path.join(self.imgdir,
                '*.'+self.imgtype)))
        # Output directory
        if outdir is None:
            self.outdir = os.path.join(datadir, 'Output')
        else:
            self.outdir = outdir
        if not os.path.isdir(self.outdir):
            print('Making output directory: \n', self.outdir)
            os.mkdir(self.outdir)
        # Image grid directory
        if griddir is None:
            self.griddir = os.path.join(datadir, 'Image_grids')
        else:
            self.griddir = griddir
        if not os.path.isdir(self.griddir):
            print('Making output directory: \n', self.outdir)
            os.mkdir(self.griddir)
        # Image grid netCDF file
        self.im_grid_res = im_grid_res
        if gridfile is None:
            self.gridfile = os.path.join(self.griddir, 
                    'im_grids_{}cm.nc'.format(int(self.im_grid_res)))
        else:
            self.gridfile = gridfile
        # Filenames for horizon radii & angles (files must be
        # generated with self.get_horizon_angles())
        self.horizon_radii = os.path.join(self.outdir,
                'horizon_radii.txt')
        self.horizon_angles = os.path.join(self.outdir,
                'horizon_angles.txt')

    def get_horizon_angles(self, K, dist, ROI, savetxt=True,
            plot=True, fname_plot=None, x_axis=None, x_label=None,
            return_timestamps=False, ndespike=2, despike='tukey'):
        """
        Loops through all frames in self.imgdir, detects horizons
        and makes lists of the horizon radii and angles. Despiking
        is performed on the timeseries of radii and angles as an
        additional quality control. 
        Parameters:
            K - ndarray; camera intrinsic matrix
            dist - ndarray; lens distortion coefficients
            ROI - ndarray; region of interest in images
            savetxt - bool; set to True to save angles in .txt
                      files
            plot - bool; set to True to make plots of angles
            fname_plot - str; if not None, saves plots with fname
            x_axis - ndarray; x axis for plots
            x_axis - str; x axis label
            return_timestamps - bool; if True, extracts timestamps
                                from images and returns array of 
                                DT objects.
            ndespike - int; number of despiking iterations
            despike - str; name of despiking algorithm
        """
        # Check if horizon_radii file already exists
        if not os.path.isfile(self.horizon_radii):
            print('Finding horizon angles for images in %s \n'
                    % self.imgdir)
            # Initialise motion correction object
            MC = swm.MotionCorrection(K=K, dist=dist)
            # Initialize output lists
            horizon_radii = np.zeros(len(self.images))
            horizon_angles = np.zeros(len(self.images))
            horizon_inc = np.zeros(len(self.images))
            horizon_roll = np.zeros(len(self.images))
            ts = [] # raw timestamps list
            # Loop over images and find horizon
            for i,fn in enumerate(self.images):
                print('Horizon %s \n' % i)
                im = cv2.imread(fn, 0)
                if return_timestamps:
                    # Read timestamp
                    ts.append(swi.read_PtGrey_timestamp(im))
                # Undistort image
                im_undist = cv2.undistort(im, cameraMatrix=K,
                                distCoeffs=dist)
                # Find horizon
                r_hor, theta_hor, _ = swm.find_horizon(im_undist, ROI)
                # Save radii and angles
                horizon_radii[i] = r_hor
                horizon_angles[i] = theta_hor

            if return_timestamps:
                # Combine timestamps
                timestamps = swi.combine_PtGrey_timestamps(self.images,
                        raw_timestamps=ts)

            # Despike radii and angles lists/arrays
            if despike == 'tukey':
                for nd in range(ndespike):
                    horizon_angles, mask_desp_a = tukey_53h(horizon_angles,
                            otsu_k=True)
                    horizon_radii, mask_desp_r = tukey_53h(horizon_radii,
                            otsu_k=True)

            # Compute incidence and roll angles from despiked thetas and radii
            print('Calculating incidence and roll angles ... \n')
            for i in range(len(horizon_angles)):
                inc, roll = swm.horizon_to_angles(r=horizon_radii[i], 
                        theta=horizon_angles[i], K=K)
                horizon_inc[i] = inc
                horizon_roll[i] = roll

            # Save to .txt files
            if savetxt:
                np.savetxt(self.horizon_radii, horizon_radii)
                np.savetxt(self.horizon_angles, horizon_angles)
                np.savetxt(os.path.join(self.outdir,'horizon_inc.txt'),
                        horizon_inc)
                np.savetxt(os.path.join(self.outdir,'horizon_roll.txt'),
                        horizon_roll)

            # Plot angles time series if requested
            if plot:
                fig, ax = plt.subplots(nrows=3, figsize=(12,18),
                        sharex=True)
                if x_axis is None:
                    if return_timestamps:
                        x_axis = pd.to_datetime(timestamps).to_pydatetime()
                        x_label = 'Time (UTC)'
                    else:
                        # Make default x-axis for plots
                        x_axis = np.arange(len(horizon_angles))
                        x_label = 'Timestep'
                elif x_axis is not None and x_label is None:
                    x_label = ' '
                # Plot theta in degrees
                ax[0].plot(x_axis, np.degrees(horizon_angles))
                ax[0].set_ylabel(r'$\theta (^\circ)$')
                # Plot incidence angles in degrees
                ax[1].plot(x_axis, np.degrees(horizon_inc))
                ax[1].set_ylabel(r'Incidence angle $(^\circ)$')
                # Plot roll angles in degrees
                ax[2].plot(x_axis, np.degrees(horizon_roll))
                ax[2].set_xlabel(x_label)
                ax[2].set_ylabel(r'Roll angle $(^\circ)$')
                if return_timestamps:
                    # rotate and align the tick labels so they look better
                    fig.autofmt_xdate()
                    # use a more precise date string for the x axis 
                    # locations in the toolbar
                    ax[2].fmt_xdata = mdates.DateFormatter(
                            '%mmm-%d %H:%M:%S')
                plt.tight_layout()
                if fname_plot is not None:
                    plt.savefig(fname_plot)
        else:
            # Load existing radii and angles
            horizon_radii = np.loadtxt(self.horizon_radii)
            horizon_angles = np.loadtxt(self.horizon_angles)
            horizon_inc = np.loadtxt(os.path.join(self.outdir,
                'horizon_inc.txt'))
            horizon_roll = np.loadtxt(os.path.join(self.outdir,
                'horizon_roll.txt'))

        if return_timestamps:
            return horizon_radii, horizon_angles, timestamps
        else:
            return horizon_radii, horizon_angles

        
    def crop_borders(self, im_warped, crop_factor=1.0):
        """
        Calls swi.crop_borders
        """
        fignm = os.path.join(self.outdir, 'crop_borders_debug.png')
        im_cropped, (r_min, r_max, c_min, c_max) = swi.crop_borders(
                im_warped, crop_factor=crop_factor, 
                debugging_fignm=fignm)

        return im_cropped, (r_min, r_max, c_min, c_max)



    def find_common_rect(self, K, dist, ROI, r_hor=None,
            theta_hor=None, savetxt=True, plot=False,
            fname_plot=None):
        """
        Loops through all frames in self.imgdir, stabilizes them
        according to horizon line, and fits the largest-area
        rectangle to each stabilized image. Returns the corners
        of the largest rectangle that fits in each frame in the
        form (r_min, r_max, c_min, c_max) in tuple opt_corners.
        Also stores all rectangle corners in list all_corners.

        Parameters:
            K - ndarray; camera intrinsic matrix
            dist - ndarray; lens distortion parameters vector
            savetxt - bool; if True, save opt_corners to .txt
                            file in self.imgdir
            plot - bool; if True, plot smallest-area rectangle on
                   the corresponding image
            fname_plot - str; filename for saving plot
        """
        print('Finding largest common rectangle for frames in %s \n'
                % self.imgdir)
        # Initialise motion correction object
        MC = swm.MotionCorrection(K=K, dist=dist)
        # Initialize list for storing rectangle corners and areas
        all_corners = [] 
        areas = []
        # Loop through and stabilise images
        for i,fn in enumerate(self.images):
            print('{} \n'.format(i))
            im = cv2.imread(fn, 0)
            if r_hor is not None and theta_hor is not None:
                # Stabilize using despiked r_hor and theta_hor
                im_stab, im_undist = MC.stabilise_horizon(
                        im, ROI, 0, return_orientation=False,
                        r_hor=r_hor[i],
                        theta_hor=theta_hor[i],
                        )
            else:
                im_stab, im_undist = MC.stabilise_horizon(
                        im, ROI, 0, return_orientation=False)
            # Crop horizon from stabilised image
            im_stab_crop, v_hor_stab = swi.crop_horizon(im_stab,
                    MC, K)
            # Crop out black borders (and a bit more)
            _, crop_corners = self.crop_borders(im_stab_crop,
                    crop_factor=1.0)
            # Add horizon (and a bit more) to crop corners
            # Have to go around 120 pixels under the horizon to
            # avoid projection to earth coordinates from becoming
            # too warped.
            crop_corners = (
                    int(v_hor_stab[0] + 125 + crop_corners[0]),
                    int(v_hor_stab[0] + crop_corners[1]), 
                    crop_corners[2], crop_corners[3]
                    )
            all_corners.append(crop_corners)
            # Calculate area
            area = ((crop_corners[1] - crop_corners[0]) * 
                    (crop_corners[3] - crop_corners[2]))
            areas.append(area)

        # Find min and max elements of corners list
        mins = list(map(min, zip(*all_corners)))
        maxs = list(map(max, zip(*all_corners)))
        # Find common rectangle corners
        opt_corners = (maxs[0], mins[1], maxs[2], mins[3])
        if savetxt:
            np.savetxt(os.path.join(self.outdir, 'opt_corners.txt'),
                    opt_corners, delimiter=',', fmt='%i')
        if plot:
            # Find index of smallest-area rectangle
            ind_min_area = areas.index(np.min(areas))
            # Open image corresponding to minimum rectangle area
            im = cv2.imread(self.images[ind_min_area], 0)
            im_stab, im_undist = MC.stabilise_horizon(
                    im, ROI, 0, return_orientation=False)
            # Plot
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(111)
            ax.imshow(im_stab, cmap=plt.cm.gray, aspect='auto')
            # Crop box
            ax.plot((opt_corners[2],opt_corners[3]), 
                    (opt_corners[0],opt_corners[0]),'-g')
            ax.plot((opt_corners[2],opt_corners[3]), 
                    (opt_corners[1],opt_corners[1]),'-g')
            ax.plot((opt_corners[2],opt_corners[2]), 
                    (opt_corners[0],opt_corners[1]),'-g')
            ax.plot((opt_corners[3],opt_corners[3]), 
                    (opt_corners[0],opt_corners[1]),'-g')
            if fname_plot is not None:
                plt.savefig(fname_plot)
            else:
                plt.show()
            plt.close()

        return opt_corners


    def rectify_images(self, K, dist, ROI, focus_corners,
            plot_reproj=False, plot_grid=False, timestamps=None):
        """
        Loop through all images in self.imgdir and stabilize,
        rectify and grid the images. The rectified image grids are
        saved into a netcdf file along with other variables such as
        the x,y grids and grid mask.

        Parameters:
            K - camera intrinsic matrix
            dist - radial distortion vector
            ROI - region of interest for cropping out part of
                  images
            focus_corners - corners of focus area rectangle
            plot_reproj - bool; Set to True to make plots of reprojected
                          images.
            plot_grid - bool; Set to True to make plot of grid
                        resolution.
            timestamps - DT ndarray; array of timestamps for the
                         images.
        """
        # Turn interactive plotting off
        plt.ioff()

        # Initialise motion correction object
        MC = swm.MotionCorrection(K=K, dist=dist)

        #  Define grid resolution in meters (originally in cm)
        dx = dy = self.im_grid_res / 100

        # Output directory for figures
        if plot_reproj:
            res = int(self.im_grid_res) # resolution for filename
            figdir = os.path.join(self.outdir,
                    'Reprojected_images_{}cm'.format(res))
            # Make output directory if it does not exist
            if not os.path.isdir(figdir):
                os.mkdir(figdir)

        # Initialize netCDF file
        nc_name = os.path.join(self.griddir, self.gridfile)
        nc_out = nc.Dataset(nc_name, 'w')
        # Get image timestamps as DT objects
        if timestamps is None:
            timestamps = swi.combine_PtGrey_timestamps(self.images)
        print('First timestamp: {} \n'.format(timestamps[0]))
        unout = 'seconds since {}'.format(
                DT.strftime(timestamps[0], '%Y-%m-%d %H:%M:%S'))
        ngrids = len(self.images)
        nc_out.createDimension('time', ngrids);
        timevar = nc_out.createVariable('time', 'float64', ('time'))
        timevar.setncattr('units', unout)
        timevar[:] = nc.date2num(timestamps, unout)

        # Loop through and stabilise images
        for i,fn in enumerate(self.images):
            print(' %s \n' % i)
            if i==0:
                # Read and stabilize first image in directory
                im = cv2.imread(fn, 0)
                im_stab, im_undist, _, r_max, theta_max, inc, roll = MC.stabilise_horizon(
                        im, ROI, 0)
                # Define cropped ROI from focus_corners
                im_stab_box = im_stab[focus_corners[0]:focus_corners[1],
                            focus_corners[2]:focus_corners[3]]
                # Project stabilized rectangle to earth coor.
                x_box_stab, y_box_stab = MC.image_to_world(
                        u=(focus_corners[2], focus_corners[3]), 
                        v=(focus_corners[0], focus_corners[1]),
                        inc=MC.incStab, roll=MC.rollStab, azi=0)
                # Interpolate stabilized image within im_stab_box to regular
                # 0.8x0.8 m grid, and plot original x_box_stab,
                # y_box_stab resolution
                fname_grid = os.path.join(self.outdir, 
                        'rectified_box_resolution.png')
                # Make output x,y grid
                xlim = (np.ceil(np.min(x_box_stab)),
                        np.floor(np.max(x_box_stab)))
                ylim = (np.ceil(np.min(y_box_stab)), 
                        np.floor(np.max(y_box_stab)))
                y_grid, x_grid = np.mgrid[ylim[0]:ylim[1]:dy, xlim[0]:xlim[1]:dx]
                y_grid = np.flip(y_grid) # "Flip" y_grid
                # Speed up gridding by calculating vertices and weights 
                # only once.
                xy = np.vstack(
                        (x_box_stab.flatten(),y_box_stab.flatten())).T 
                xy_grid = np.vstack(
                        (x_grid.flatten(),y_grid.flatten())).T
                vertices, weights = swg.interp_weights(xy, xy_grid)
                im_grid, x_grid, y_grid, im_mask = swi.img_to_grid(
                        im_stab_box, X=x_box_stab, Y=y_box_stab,
                        x_grid=x_grid, y_grid=y_grid,
                        dx=dx, dy=dy, plot_grid=plot_grid,
                        fname_grid=fname_grid, vertices=vertices,
                        weights=weights)
#                print('close? ', np.allclose(im_grid, im_grid2,
#                    equal_nan=True))
                # Define nc dimensions
                ny, nx = x_grid.shape
                x = x_grid[0,:]
                y = y_grid[:,0]
                nc_out.createDimension('x',nx);
                nc_out.createDimension('y',ny);
                # Create nc variables
                x_var = nc_out.createVariable('x','float32',('x'))
                x_var[:] = x
                y_var = nc_out.createVariable('y','float32',('y'))
                y_var[:] = y
                # Image grids
                imgrid_var = nc_out.createVariable('im_grid','uint8',
                        ('time','y','x'))
                xgrid_var = nc_out.createVariable('x_grid','float32',
                        ('y','x'))
                xgrid_var.setncattr('units','m') 
                xgrid_var[:,:] = x_grid
                ygrid_var = nc_out.createVariable('y_grid','float32',
                        ('y','x'))
                ygrid_var.setncattr('units','m')
                ygrid_var[:,:] = y_grid
                # Mask
                mask_var = nc_out.createVariable('mask', 'uint8', ('y','x'))
                mask_var[:,:] = im_mask
            else:
                im = cv2.imread(fn, 0)
                im_stab, im_undist, _, r_max, theta_max, inc, roll = MC.stabilise_horizon(
                        im, ROI, 0, )
                # Define cropped ROI from focus_corners
                im_stab_box = im_stab[focus_corners[0]:focus_corners[1],
                            focus_corners[2]:focus_corners[3]]
                im_grid, _, _ = swi.img_to_grid(im_stab_box, 
                        X=x_box_stab, Y=y_box_stab, dx=dx, dy=dy,
                        x_grid=x_grid, y_grid=y_grid,
                        return_mask=False, vertices=vertices,
                        weights=weights)

            # Save im_grid to nc file
            im_grid = np.array(im_grid, dtype=np.uint8)
            imgrid_var[i,:,:] = im_grid

            # Make plots if requested, but only for first 100
            # frames
            if i < 100:
                if plot_reproj:
                    print('Plotting reprojected image ... \n')
                    # Get horizon line for plotting
                    _, v_hor_stab = swi.crop_horizon(im_stab, MC, K)
                    # Reproject crop box to undistorted image
                    u_box_undist, v_box_undist = swm.reproject_coordinates(
                            projection='image2image',
                            inc=MC.incStab, roll=MC.rollStab, azi=MC.aziStab,
                            inc2=inc, roll2=roll, azi2=0, K=MC.K, H=MC.H, 
                            u=(focus_corners[2], focus_corners[3]), 
                            v=(focus_corners[0], focus_corners[1]))

                    fig, ax = plt.subplots(ncols=3, figsize=(6.5,2.6))
                    # Plot undistorted image
                    ax[0].imshow(im_undist, cmap=plt.cm.gray, aspect='auto')
                    # Plot edges of warped box
                    # Left edge, up-down
                    ax[0].plot(u_box_undist[:,1],v_box_undist[:,1],'-g')
                    # Right edge, up-down
                    ax[0].plot(u_box_undist[:,-1],v_box_undist[:,-1],'-g')
                    # Upper edge, left-right
                    ax[0].plot(u_box_undist[1,:],v_box_undist[1,:],'-g')
                    # Lower edge, left-right
                    ax[0].plot(u_box_undist[-1,:],v_box_undist[-1,:],'-g')
                    ax[0].set_title('Undistorted')
                    # Plot horizon in undistorted image
                    u_hor = np.array([0,im_undist.shape[1]-1])
                    v_hor = swm.horizon_pixels(r=r_max, theta=theta_max,
                            u=u_hor)
                    ax[0].plot(u_hor, v_hor, color='r', linestyle='--')
                    ax[0].set_xlabel('u')
                    ax[0].set_ylabel('v')
                    ax[0].annotate('(a)', xy=(0.1, 0.08), color='white',
                            xycoords='axes fraction', fontsize=12)
                    # Plot stabilised image
                    u_hor_stab = (0, im_undist.shape[1]-1)
                    ax[1].imshow(im_stab, cmap=plt.cm.gray, aspect='auto')
                    # Crop box
                    ax[1].plot((focus_corners[2],focus_corners[3]), 
                            (focus_corners[0],focus_corners[0]),'-g')
                    ax[1].plot((focus_corners[2],focus_corners[3]), 
                            (focus_corners[1],focus_corners[1]),'-g')
                    ax[1].plot((focus_corners[2],focus_corners[2]), 
                            (focus_corners[0],focus_corners[1]),'-g')
                    ax[1].plot((focus_corners[3],focus_corners[3]), 
                            (focus_corners[0],focus_corners[1]),'-g')
                    ax[1].set_title('Stabilized')
                    # Horizon
                    ax[1].plot(u_hor_stab, v_hor_stab, color='r',
                            linestyle='--')
                    ax[1].set_xlabel('u')
                    ax[1].set_ylabel('v')
                    ax[1].annotate('(b)', xy=(0.1, 0.08), color='white',
                            xycoords='axes fraction', fontsize=12)
                    # Plot reprojected (and gridded) image within 
                    # stabilized rectangle
                    im_grid_plot = np.ma.masked_where(im_grid==0, im_grid)
                    ax[2].pcolormesh(x_grid, y_grid, im_grid_plot,
                            cmap=plt.cm.gray, vmin=0, vmax=255,
                            rasterized=True)
                    # Stabilized rectangle in green
                    ax[2].plot(x_box_stab[:,1], y_box_stab[:,1],'-g')
                    ax[2].plot(x_box_stab[:,-1], y_box_stab[:,-1],'-g')
                    #ax[2].plot(x_box_stab[1,:], y_box_stab[1,:]+1,'-g')
                    ax[2].plot(x_box_stab[1,:], y_box_stab[1,:],'-g')
                    ax[2].plot(x_box_stab[-1,:], y_box_stab[-1,:],'-g')
                    ax[2].set_xlim((x_box_stab.min(),x_box_stab.max()))
                    ax[2].set_ylim((y_box_stab.min(),y_box_stab.max()))
                    ax[2].set_title('Gridded')
                    ax[2].set_xlabel('m')
                    ax[2].set_ylabel('m')
                    ax[2].annotate('(c)', xy=(0.1, 0.08), color='k',
                            xycoords='axes fraction', fontsize=12)
                    for a in ax:
                        a.xaxis.set_tick_params(which='major', size=7, 
                                width=2, direction='in', top='on')
                        a.xaxis.set_tick_params(which='minor', size=4, 
                                width=1, direction='in', top='on')
                        a.yaxis.set_tick_params(which='major', size=7, 
                                width=2, direction='in', right='on')
                        a.yaxis.set_tick_params(which='minor', size=4, 
                                width=1, direction='in', right='on')
                    plt.tight_layout()
                    # Save figure
                    plt.savefig(os.path.join(figdir, 
                        'reprojected_images_{:03d}.pdf'.format(i)), 
                        dpi=300, transparent=False, bbox_inches='tight')
                    plt.close(fig)

        # Close netCDF file
        nc_out.close()

    
    def threshold_KM11(self, DS, method='global', smooth=False,
            bins=range(0,256), d=7, sigmaColor=75, sigmaSpace=75,
            plot=False, n_plot=50, figname=None, hist=None,
            x_grid=None, y_grid=None, mask=None, correct_background=False,
            focus_thresh=None):
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
            figname - str; figure name for saving plot in self.outdir
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
            focus_thresh - int; threshold to focus plotting on.
        Returns:
            wc_cov_20 - whitecap fraction based on threshold.
        """
        # Define mask and x,y grids
        if x_grid is None:
            x_grid = DS.x_grid.values
        if y_grid is None:
            y_grid = DS.y_grid.values
        if mask is None:
            mask = DS.mask.values

        if correct_background:
            # Plot name for background row mean pixel intensity figure
            png_bg = os.path.join(self.outdir, 'mean_background.png')
            images = DS.im_grid.where(DS.mask != 0) # Mask image grids
            # Remove background pixel intensity gradient
            images = swi.remove_background_gradients(images, mask=mask,
                    cv_mask=True, plot=True, figname=png_bg,
                    xgrid=DS.x_grid.values, ygrid=DS.y_grid.values)
        else:
            images = DS.im_grid.where(DS.mask != 0).values # Mask image grids


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
            # Save histogram
            if correct_background:
                fn_hist = os.path.join(self.outdir, 'hist_bgcorr.txt') # filename
            else:
                fn_hist = os.path.join(self.outdir, 'hist.txt') # filename
            if not os.path.isfile(fn_hist):
                print('Saving histogram ... \n')
                np.savetxt(fn_hist, hist)

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
            # There might be smaller bumps beyond the main peak; take the last
            # point at which L_pp goes below the threshold
            thresh_i = np.atleast_1d(peak_ind + np.squeeze(np.where(
                L_pp[peak_ind:]<th*peak_value))[-1])
            thresh_dict['thresh_{}'.format(int(round(th*100)))] = int(thresh_i)
        # Save dict as .json
        print('Saving thresholds to .json ... \n')
        fname_json = os.path.join(self.outdir,
                'thresholds_{}cm_rescaled.json'.format(int(self.im_grid_res)))
        with open(fname_json, 'w') as fp:
            json.dump(thresh_dict, fp)

        if plot:
            if focus_thresh is None:
                thresh_list = [5, 10, 20] # Default thresh's to plot
            else:
                thresh_list = [
                        focus_thresh, 
                        focus_thresh + focus_thresh//2, 
                        focus_thresh + 2*focus_thresh//2
                        ]
            self.plot_thresholds(DS, hist, L_pp, thresh_dict, n_plot, 
                    thresholds=thresh_list, figname=figname)

        return thresh_dict


    def compute_W_tot(self, DS, thresh_dict, thresh_str, plot=True,
            correct_background=False):
        """
        Calculates total (active + stage B) whitecap coverage 
        fraction on input images based on the brightness intensity
        threshold thresh (an integer between 0 and 255).
        Save timeseries of frame-wise W in netcdf file.

        Parameters:
            thresh - int; brightness threshold (0-255)
            DS - xarray dataset; output of self.rectify_images()
            plot - bool; set to True to generate plot of W as timeseries
        """
        thresh = thresh_dict[thresh_str]
        # Mask image grids according to mask
        print('Masking image grids ... \n')
        images = DS.im_grid.where(DS.mask != 0)
        # Average brightness of all grids
#        im_mean = np.floor(images.mean(skipna=True).values)
        # Make xr.DataArray for W
        da = xr.DataArray(np.zeros(len(images)), coords=[DS.time.values],
            dims=['time'])
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
        if correct_background:
            nc_name = os.path.join(self.outdir, 'w_tot_{0}cm_{1}_{2}.nc'.format(
                int(self.im_grid_res), thresh_str, 'bgcorr'))
            png_name = os.path.join(self.outdir, 'w_tot_{0}cm_{1}_{2}.png'.format(
                int(self.im_grid_res), thresh_str, 'bgcorr'))
        else:
            nc_name = os.path.join(self.outdir, 'w_tot_{0}cm_{1}.nc'.format(
                int(self.im_grid_res), thresh_str))
            png_name = os.path.join(self.outdir, 'w_tot_{0}cm_{1}.png'.format(
                int(self.im_grid_res), thresh_str))
        ds.to_netcdf(nc_name)

        # Plot if requested
        if plot:
            # Read netcdf file
            fig = plt.figure()
            ax = fig.add_subplot(111)
            #ax.plot(pd.to_datetime(DS.time.values), W)
            ds.w_tot.plot(ax=ax)
            plt.tight_layout()
            plt.savefig(png_name)
            plt.close()


    def plot_thresholds(self, DS, hist, L_pp, thresh_dict, n_plot=50, 
            thresholds=None, figname=None):
        """
        Generate plots of histogram, L_pp, raw image grid and 3 thresholded 
        image grids at different percentage thresholds (KM11 thresholding).
        """
        if thresholds is None:
            thresholds = [5, 10, 20] # default thresholds list
        x_grid = DS.x_grid.values
        y_grid = DS.y_grid.values
        mask = DS.mask.values
        # Define thresholds
        thresh_1 = thresh_dict['thresh_{}'.format(int(thresholds[0]))]
        thresh_2 = thresh_dict['thresh_{}'.format(int(thresholds[1]))]
        thresh_3 = thresh_dict['thresh_{}'.format(int(thresholds[2]))]
        # Make range array from 0:n_plot plus n_plot random numbers
        plot_range = np.arange(2 * n_plot)
        rand = np.arange(n_plot, len(DS.im_grid))
        np.random.shuffle(rand)
        rand_subset = rand[:n_plot]
        rand_subset.sort()
        plot_range[n_plot:] = rand_subset
        print('Plotting thresholded images ... \n')
        for i, im in enumerate(DS.im_grid[plot_range]):
            if i % 10 == 0:
                print('Thresh {} \n'.format(i))
            # Plot histogram, L_pp and example thresholded images
            fig, ax = plt.subplots(nrows=3, ncols=2, 
                    figsize=(10,10))
            ax[0,0].plot(hist)
            ax[0,0].axvline(thresh_1, c='r', 
                    label='{}%'.format(int(thresholds[0])),
                    linestyle='-')
            ax[0,0].axvline(thresh_2, c='g',
                    label='{}%'.format(int(thresholds[1])),
                    linestyle='--')
            ax[0,0].axvline(thresh_3, c='b', 
                    label='{}%'.format(int(thresholds[2])),
                    linestyle=':')
            ax[0,0].set_xlim([0, 256])
            ax[0,0].set_title('Pixel intensity histogram')
            ax[0,0].legend()
            ax[0,1].semilogx(L_pp)
            ax[0,1].semilogx(thresh_1, L_pp[thresh_1], 'ro',
                    label='{}%'.format(int(thresholds[0])))
            ax[0,1].semilogx(thresh_2, L_pp[thresh_2], 'g^',
                    label='{}%'.format(int(thresholds[1])))
            ax[0,1].semilogx(thresh_3, L_pp[thresh_3], 'bs',
                    label='{}%'.format(int(thresholds[2])))
            ax[0,1].set_title('$L_{pp}$', usetex=True)
            ax[0,1].legend()
            # Plot thresholded image grids
            im = np.ma.masked_where(mask==0,im)
            ax[1,0].pcolormesh(x_grid, y_grid, im,
                    cmap=plt.cm.gray, vmin=0, vmax=255)
            ax[1,0].set_xlabel('m')
            ax[1,0].set_ylabel('m')
            ax[1,1].pcolormesh(x_grid, y_grid, im>thresh_1,
                    cmap=plt.cm.gray, vmin=0, vmax=1)
            ax[1,1].set_xlabel('m')
            ax[1,1].set_ylabel('m')
            ax[2,0].pcolormesh(x_grid, y_grid, im>thresh_2,
                    cmap=plt.cm.gray, vmin=0, vmax=1)
            ax[2,0].set_xlabel('m')
            ax[2,0].set_ylabel('m')
            ax[2,1].pcolormesh(x_grid, y_grid, im>thresh_3, 
                    cmap=plt.cm.gray, vmin=0, vmax=1)
            ax[2,1].set_xlabel('m')
            ax[2,1].set_ylabel('m')
            # Compute whitecap coverage for example plots
            wc_cov_1 = (im>thresh_1).sum() / im.count()
            wc_cov_2 = (im>thresh_2).sum() / im.count()
            wc_cov_3 = (im>thresh_3).sum() / im.count()
            ax[1,0].set_title('Stabilized ROI')
            ax[1,1].set_title('{}% threshold, W = {:02f}'.format(
                thresholds[0], wc_cov_1))
            ax[2,0].set_title('{}% threshold, W = {:02f}'.format(
                thresholds[1], wc_cov_2))
            ax[2,1].set_title('{}% threshold, W = {:02f}'.format(
                thresholds[2], wc_cov_3))
            # Use timestamp as overall figure title
            supt = pd.to_datetime(DS.time[plot_range[i]].values).strftime(
                    '%Y-%m-%d %H:%M:%S')
            plt.suptitle(supt)
            # Set equal aspect ratios
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            if figname is not None:
                outdir = os.path.join(self.outdir,
                        'Thresholded_images_{}cm'.format(int(self.im_grid_res)))
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                figname_i = (
                        figname.split('.')[0]+'_{:04d}'.format(plot_range[i]) +
                        '.png')
                plt.savefig(os.path.join(outdir, figname_i))
            else:
                plt.show()
            plt.close()

    def plot_ST15_reproj(self, MC, K, im_undist, im_stab, inc, roll,
            r_max, theta_max, figname):
        """
        Makes 3-panel plot of horizon-based image stabilization process
        of Schwendeman and Thomson (2015, JTECH).
        """
        print('Plotting reprojected image ... \n')
        # Get horizon line for plotting
        _, v_hor_stab = swi.crop_horizon(im_stab, MC, K)
        # Reproject crop box to undistorted image
        u_box_undist, v_box_undist = swm.reproject_coordinates(
                projection='image2image',
                inc=MC.incStab, roll=MC.rollStab, azi=MC.aziStab,
                inc2=inc, roll2=roll, azi2=0, K=MC.K, H=MC.H, 
                u=(focus_corners[2], focus_corners[3]), 
                v=(focus_corners[0], focus_corners[1]))

        fig, ax = plt.subplots(ncols=3, figsize=(6.5,3.5))
        # Plot undistorted image
        ax[0].imshow(im_undist, cmap=plt.cm.gray, aspect='auto')
        # Plot edges of warped box
        # Left edge, up-down
        ax[0].plot(u_box_undist[:,1],v_box_undist[:,1],'-g')
        # Right edge, up-down
        ax[0].plot(u_box_undist[:,-1],v_box_undist[:,-1],'-g')
        # Upper edge, left-right
        ax[0].plot(u_box_undist[1,:],v_box_undist[1,:],'-g')
        # Lower edge, left-right
        ax[0].plot(u_box_undist[-1,:],v_box_undist[-1,:],'-g')
        ax[0].set_title('Undistorted')
        # Plot horizon in undistorted image
        u_hor = np.array([0,im_undist.shape[1]-1])
        v_hor = swm.horizon_pixels(r=r_max, theta=theta_max,
                u=u_hor)
        ax[0].plot(u_hor, v_hor, color='r', linestyle='--')
        ax[0].annotate('(a)', xy=(0.1, 0.1),
                xycoords='axes fraction', fontsize=12)
        # Plot stabilised image
        u_hor_stab = (0, im_undist.shape[1]-1)
        ax[1].imshow(im_stab, cmap=plt.cm.gray, aspect='auto')
        # Crop box
        ax[1].plot((focus_corners[2],focus_corners[3]), 
                (focus_corners[0],focus_corners[0]),'-g')
        ax[1].plot((focus_corners[2],focus_corners[3]), 
                (focus_corners[1],focus_corners[1]),'-g')
        ax[1].plot((focus_corners[2],focus_corners[2]), 
                (focus_corners[0],focus_corners[1]),'-g')
        ax[1].plot((focus_corners[3],focus_corners[3]), 
                (focus_corners[0],focus_corners[1]),'-g')
        ax[1].set_title('Stabilized')
        # Horizon
        ax[1].plot(u_hor_stab, v_hor_stab, color='r',
                linestyle='--')
        ax[1].annotate('(b)', xy=(0.1, 0.1),
                xycoords='axes fraction', fontsize=12)
        # Plot reprojected (and gridded) image within 
        # stabilized rectangle
        ax[2].pcolormesh(x_grid, y_grid, im_grid,
                cmap=plt.cm.gray, vmin=0, vmax=255)
        # Stabilized rectangle in green
        ax[2].plot(x_box_stab[:,1], y_box_stab[:,1],'-g')
        ax[2].plot(x_box_stab[:,-1], y_box_stab[:,-1],'-g')
        ax[2].plot(x_box_stab[1,:], y_box_stab[1,:]+1,'-g')
        ax[2].plot(x_box_stab[-1,:], y_box_stab[-1,:],'-g')
        ax[2].set_xlim((x_box_stab.min(),x_box_stab.max()))
        ax[2].set_ylim((y_box_stab.min(),y_box_stab.max()))
        ax[2].set_title('Gridded')
        ax[2].set_xlabel('m')
        ax[2].set_ylabel('m')
        ax[2].annotate('(c)', xy=(0.1, 0.1),
                xycoords='axes fraction', fontsize=12)
        for a in ax:
            a.xaxis.set_tick_params(which='major', size=7, 
                    width=2, direction='in', top='on')
            a.xaxis.set_tick_params(which='minor', size=4, 
                    width=1, direction='in', top='on')
            a.yaxis.set_tick_params(which='major', size=7, 
                    width=2, direction='in', right='on')
            a.yaxis.set_tick_params(which='minor', size=4, 
                    width=1, direction='in', right='on')
        plt.tight_layout()
        # Save figure
        plt.savefig(figname, dpi=300, transparent=False, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    """
    Main script.
    """
    from argparse import ArgumentParser

    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-dr", 
                help=("Path to data_root (work) directory"),
                type=str,
                default='/home/mikapm/stereo-wave/WhiteCapFraction/Test_case',
                )
        parser.add_argument("-imgdir", 
                help=("Path to image directory"),
                type=str,
                default=('/home/mikapm/stereo-wave/WhiteCapFraction/Images/'
                    '20191210_1817UTC/'),
                )
        parser.add_argument("-calibdir", 
                help=("Path to calibration files directory"),
                type=str,
                )
        parser.add_argument("-outdir", 
                help=("Path to output figures directory"),
                type=str,
                default=('/home/mikapm/stereo-wave/WhiteCapFraction/Figures/whitecaps_out'),
                )
        parser.add_argument("-intype", 
                help=("Input image format (file extension)"),
                type=str,
                choices=['tif', 'png', 'jpg', 'pgm'],
                default='pgm',
                )
        parser.add_argument("-outtype", 
                help=("Output image format (file extension)"),
                type=str,
                choices=['tif', 'png', 'jpg', 'pgm'],
                default='tif',
                )
        parser.add_argument("-res", 
                help=("Image grid resolution in cm (!)"),
                type=float,
                default=80,
                )
        parser.add_argument("-thresh", 
                help=("Percentage threshold to use (following KM11)"),
                type=float,
                default=5,
                )
        parser.add_argument("--plot_reproj", 
                help=("Make plots of reprojected images?"),
                action="store_false", # True by default
                )
        parser.add_argument("--plot_grid", 
                help=("Plot stabilized image grid?"),
                action="store_false",
                )
        parser.add_argument("--plot_rect", 
                help=("Plot minimum area rectangle?"),
                action="store_false",
                )
        parser.add_argument("--plot_hor", 
                help=("Plot detected horizon angles?"),
                action="store_false",
                )
        parser.add_argument("--plot_thresh", 
                help=("Plot thresholded image(s)?"),
                action="store_false",
                )
        parser.add_argument("--plot_wtot", 
                help=("Plot thresholded image(s)?"),
                action="store_false",
                )
        parser.add_argument("--replot_thresh", 
                help=("Remake plots of thresholded image(s)?"),
                action="store_true", # False by default
                )
        parser.add_argument("--smooth_thresh", 
                help=("Smooth image(s) before thresholding?"),
                action="store_true",
                )
        parser.add_argument("--correct_background", 
                help=("Remove background gradients"),
                action="store_true",
                )
        return parser.parse_args(**kwargs)

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

    # Initialise default WC object
    WC = WhiteCapFraction(datadir=args.dr, im_grid_res=args.res,
            imgdir=args.imgdir)
    
    print('Processing ', WC.datadir)

    if not os.path.isfile(WC.gridfile):

        # Read intrinsic parameters from correct .mat file
        if args.calibdir is None:
            calibdir = os.path.join(args.dr, 'CalibrationFiles')
        else:
            calibdir = args.calibdir
        fn0 = WC.images[0].split('/')[-1]
        if fn0[4:6] == '83':
            cam = 'STBD'
        elif fn0[4:6] == '65':
            cam = 'PORT'
        fn_intr = os.path.join(calibdir, 'intr_{}.mat'.format(cam))
        cameraParams = spio.loadmat(fn_intr)
        K = cameraParams['K'].T # intrinsic matrix
        dist = np.zeros(5) # radial distortion params vector
        dist[0:2] = cameraParams['dist'].squeeze() # only have k1,k2

        # Read one image to get shape
        im = cv2.imread(WC.images[0], 0) # flag=0 reads image in grayscale
        nv, nu = im.shape[:2]
        if cam == 'STBD':
            ROI = [0, 120, nv, nu-50] # for cropping out casing edges
        elif cam == 'PORT':
            ROI = [0, 0, nv, nu] # Don't crop portside camera images

        # Load or find horizon angles and radii
        fname_horizon_angles = os.path.join(WC.outdir,
                'horizon_angles.png')
        # Check for .nc files with different resolutions for timestamps
        nc_files = glob.glob(os.path.join(WC.griddir, '*.nc'))
        if len(nc_files) == 0:
            # Return timestamps from get_horizon_angles()
            #horizon_radii, horizon_angles, timestamps = WC.get_horizon_angles(
            horizon_radii, horizon_angles = WC.get_horizon_angles(
                    K, dist, ROI, plot=args.plot_hor,
                    fname_plot=fname_horizon_angles,)
                    #return_timestamps=True)
        else:
            # Get timestamps from available nc file
            ds_temp = xr.open_dataset(nc_files[0], chunks={'time':10})
            timestamps = pd.to_datetime(ds_temp.time.data).to_pydatetime()
    #        else:
    #            horizon_radii = np.loadtxt(WC.horizon_radii)
    #            horizon_angles = np.loadtxt(WC.horizon_angles)
        # Find largest common rectangle in input images (after
        # stabilizing)
        opt_corners_path = os.path.join(WC.outdir, 'opt_corners.txt')
        if os.path.isfile(opt_corners_path):
            print('Loading saved focus corners ... \n')
            focus_corners = np.loadtxt(opt_corners_path, dtype=int)
        else:
            # Filename for plot of smallest rectangle
            fname_corners_plot = os.path.join(WC.outdir, 
                    'min_area_bbox.png')
            focus_corners = WC.find_common_rect(K, dist, ROI,
                    r_hor=horizon_radii, theta_hor=horizon_angles,
                    plot=args.plot_rect, fname_plot=fname_corners_plot,
                    )
        
        # Rectify and grid images
    #if not os.path.isfile(WC.gridfile):
        #print('Grid netCDF file not found, gridding images ... \n')
        print('Gridding images ... \n')
        WC.rectify_images(K, dist, ROI, focus_corners, 
                plot_reproj=args.plot_reproj, plot_grid=args.plot_grid,
                #timestamps=timestamps,
                )

    # Load image grid dataset
    print('Loading image grid dataset from netcdf file ... \n')
    DS = xr.open_dataset(WC.gridfile, chunks={'time':1})
    # Check if thresholds .json file already exists
    thresh_json = os.path.join(WC.outdir, 'thresholds_{}cm.json'.format(
        int(WC.im_grid_res)))
    if not os.path.isfile(thresh_json) or args.replot_thresh:
        print('Performing whitecap thresholding ... \n')
        # Threshold rectified image grids
        if args.correct_background:
            plotname_thresh = 'thresh_example_{}cm_thresh_{}_bgcorr.png'.format(
                    int(args.res), int(args.thresh))
            hist_txt = os.path.join(WC.outdir, 'hist_bgcorr.txt')
        else:
            plotname_thresh = 'thresh_example_{}cm_thresh_{}.png'.format(
                    int(args.res), int(args.thresh))
            hist_txt = os.path.join(WC.outdir, 'hist.txt')
        # Check if histogram is already computed
        if os.path.isfile(hist_txt):
            hist = np.loadtxt(hist_txt)
        else:
            hist = None
        # Perform thresholding. 1-50 % thresholds are output in thresh_dict
        if not args.replot_thresh:
            thresh_dict = WC.threshold_KM11(DS=DS, smooth=args.smooth_thresh,
                    hist=hist, plot=args.replot_thresh, figname=plotname_thresh,
                    correct_background=args.correct_background,
                    focus_thresh=args.thresh)
        else:
            print('Plot thresh? ', args.plot_thresh)
            thresh_dict = WC.threshold_KM11(DS=DS, smooth=args.smooth_thresh,
                    #hist=hist, plot=args.plot_thresh, figname=plotname_thresh,
                    hist=None, plot=args.plot_thresh, figname=plotname_thresh,
                    correct_background=args.correct_background,
                    focus_thresh=args.thresh)
    else:
        print('Found thresholds .json file, loading that ... \n')
        # Load .json file
        with open(thresh_json) as f:
            thresh_dict = json.load(f)
    # Use the chosen threshold
    thresh_str = 'thresh_{}'.format(int(args.thresh))

    if args.correct_background:
        w_tot_nc = os.path.join(WC.outdir, 'w_tot_{0}cm_{1}_{2}.nc'.format(
                int(WC.im_grid_res), thresh_str, 'bgcorr'))
    else:
        w_tot_nc = os.path.join(WC.outdir, 'w_tot_{0}cm_{1}.nc'.format(
                int(WC.im_grid_res), thresh_str))
    if not os.path.isfile(w_tot_nc):
        # Compute total whitecap coverage and save to netcdf
        WC.compute_W_tot(DS, thresh_dict, thresh_str, plot=args.plot_wtot,
                correct_background=args.correct_background)
    # Read w_tot netcdf file
    print('Loading w_tot dataset from netcdf file ... \n')
    ds_w = xr.open_dataset(w_tot_nc)


