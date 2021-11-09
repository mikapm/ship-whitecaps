#!/usr/bin/env python3
"""
Functions to stabilise single-camera images
from a moving vessel using the horizon tracking 
method of Schwendeman and Thomson (2015, JTECH).

Citation: Schwendeman, M., J. Thomson, 2015: 
          "A Horizon-tracking Method for Shipboard Video
          Stabilization and Rectification."  
          J. Atmos. Ocean. Tech.

This python code is based on the Matlab codes
written by Michael Schwendeman and published online at
https://github.com/mikeschwendy/HorizonStabilization
and
https://digital.lib.washington.edu/researchworks/handle/1773/38314

These functions rely on the skimage and OpenCV (cv2) image processing
libraries.
"""

import numpy as np
import os
import sys
import glob
import cv2
from math import sin, cos, tan, atan 
import skimage
from skimage.feature import canny
from skimage import transform as tf
import scipy.io as spio
import matplotlib.pyplot as plt

def angles_to_horizon(inc, roll, K):
    """
    Calculate the location of the horizon based on the 
    camera incidence and roll.

    Parameters:
        - inc and roll can be scalars, vectors or arrays and 
          must be of the same size.  
        - K is the 3x3 upper-triangular camera intrinsic matrix. 
    Returns:
        the horizon line parameters theta (in radians) 
        and r (in pixels), from the camera incidence angle, 
        inc, and roll angle, roll

    Based on Angles2Horizon.m by Michael Schwendeman, 2014.
    """

    # Intrinsic parameters from K
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    # Calculate r, theta
    theta = atan(-fx / (fy * tan(roll)))
    r = (fx * sin(roll) * cos(theta) - 
            fy * cos(roll) * sin(theta) / tan(inc) + 
            cx * cos(theta) + cy * sin(theta))
    #r = np.abs(np.atleast_1d(r))
    r = np.atleast_1d(r)
    theta = np.atleast_1d(theta)

    return r, theta

def horizon_to_angles(r, theta, K):
    """
    Calculate camera incidence and roll based on the location
    of the horizon in the image.

    Parameters:
        - theta and r can be scalars, vectors or arrays and 
          must be of the same size.  
        - K is the 3x3 upper-triangular camera intrinsic matrix.
    Returns:
        Camera incidence angle, inc, and roll angle, roll (same
        size as inputs theta and r).

    Based on Horizon2Angles.m by Michael Schwendeman, 2014.
    """
    # Intrinsic parameters from K
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    # Calculate roll and inclination angles
    roll = atan(-fx / (fy * tan(theta)))
    inc = atan((fx * sin(roll) * cos(theta) - fy *
        cos(roll) * sin(theta)) / 
        (r - cx * cos(theta) - cy * sin(theta)))
    # Correct for negative inclination angles
    inc = np.atleast_1d(inc)
    roll = np.atleast_1d(roll)
    inc[inc<=0] += np.pi 

    return inc, roll

def horizon_pixels(r, theta, u):
    """
    Parametric representation of a line: 
    r = u*cos(theta) + v*sin(theta)
    =>
    v = (r - u*cos(theta)) / sin(theta)

    Returns:
        The row coordinates v of the horizon line 
        parameters theta (in radians) and r (in pixels)
        at the column coordinates u.

    Based on HorizonPixels.m by Michael Schwendeman, 2014.

    """
    return np.abs((r - u * cos(theta)) / sin(theta))

def find_horizon(im_undist, ROI=None, method='canny', sigma=0.4):
    """
    Find horizon line in undistorted image.

    It was found during testing that images with large, bright foam
    patches made the original horizon detection method (using only
    a dilated image for the Canny edge detection) fail. Therefore,
    we added an Otsu thresholding step before the edge detection.
    This allows a much more robust horizon detection.

    Automatic Canny edge detection thresholding using median
    of pixel intensities borrowed from:
    https://www.pyimagesearch.com/2015/04/06/zero-parameter-
    automatic-canny-edge-detection-with-python-and-opencv/

    Based on FindHorizonAlternate.m by Michael Schwendeman, 2014.
    """
    # Crop out ROI if specified
    if ROI is not None:
        im_undist = im_undist[ROI[0]:ROI[2], ROI[1]:ROI[3]]
    # Erode and dilate image to accentuate gradient of horizon line
    im_erode = cv2.erode(im_undist, (5,5), iterations=1)
    im_dilate = cv2.dilate(im_erode, (5,5), iterations=1)
    # Use Otsu's threshold to separate sky and water
    ret3, im_otsu = cv2.threshold(im_dilate, 1, 255, 
            cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Compute the median of the single channel pixel intensities
    v = np.median(im_otsu)
    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # Find edges using Canny edge detection algorithm
    #edges = cv2.Canny(im_dilate, lower, upper)
    edges = cv2.Canny(im_otsu, 0, 1)

    # Find connected lines using Hough transform at 0.5 deg
    # precision
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    # hspace - Hough transform accumulator array
    # angles - Angles at which the transform is computed, in radians
    # rhos - Distance values 
    hspace, angles, rhos = tf.hough_line(
            edges, theta=tested_angles)
    # Find the two largest peaks in the Hough transform
    hpeaks, angles, rhos = tf.hough_line_peaks(
            hspace, angles, rhos, num_peaks=2)

    # r_max and theta_max are given by the highest peaks
    r_max = np.abs(rhos[0]) # take abs() to avoid negative r
    theta_max = angles[0]
    if theta_max < 0:
        theta_max += np.pi

    return r_max, theta_max, edges

def rotation_matrix(inc, roll, azi):
    """
    Returns 3D rotation matrix from input incidence angle, inc,
    roll angle, roll, and azimuth angle, azi.
    For reference see, e.g. Kleiss (2009, PhD dissert.), eq. 2.7 
    """
    R_roll = np.array([
        [cos(roll), -sin(roll), 0],
        [sin(roll), cos(roll), 0],
        [0, 0, 1]
        ])
    R_pitch = np.array([
        [1, 0, 0],
        [0, -cos(inc), -sin(inc)],
        [0, sin(inc), -cos(inc)]
        ])
    R_azi = np.array([
        [cos(azi), 0, -sin(azi)],
        [0, 1, 0],
        [sin(azi), 0, cos(azi)]
        ])

    return R_azi @ R_roll @ R_pitch 

def projection_matrix(K):
    """
    Generate 4x4 projection matrix P from input 
    3x3 matrix K. See Schwendeman & Thomson (2015), eq.2.
    """
    P = np.zeros((4,4))
    P[:3,:3] = K
    P[3,3] = 1.0

    return P

def pixel_matrix(u,v, s=1.0, d=1.0):
    """
    Generate 'pixel matrix' s[u,v,1,d].T as in eq. 1 
    of Schwendeman & Thomson (2015). Assuming s=d=1.
    """
    # Flatten pixel arrays for later
    u_flat = u.flatten()
    v_flat = v.flatten()
    # Generate matrix from pixel coordinates
    UV = np.zeros((4, u_flat.shape[0]))
    UV[0,:] = u_flat
    UV[1,:] = v_flat
    UV[2,:] = np.ones(u_flat.shape[0])
    UV[3,:] = np.ones(u_flat.shape[0]) * d

    return (UV * s)

def xy_matrix(x, y, H):
    """
    Generate world-coordinate matrix following
    Schwendeman & Thomson (2015). 
    """
    # Flatten coordinate arrays for later
    x_flat = x.flatten()
    y_flat = y.flatten()
    XY = np.zeros((4, x_flat.shape[0]))
    XY[0,:] = x_flat
    XY[1,:] = y_flat
    XY[2,:] = -np.ones(x_flat.shape[0]) * H
    XY[3,:] = np.ones(x_flat.shape[0])

    return XY


def stabilise_image(im_undist, inc1, roll1, azi1,
        inc2, roll2, azi2, K, resize_output=False):
    """
    stabilise_image() warps undistorted image im_undist
    from a camera oriented with incidence, inc1, roll, roll1, 
    and azimuth, azi1 (in radians) to a camera with constant 
    angles inc2, roll2, and azi2.  
    
    Parameters:
        inc1, roll1, and azi1 may be scalars or vectors of length P. 
        inc2, roll2, and azi2 are scalars. 
        K is the 3x3 upper-triangular camera intrinsic matrix.
        resize_output - bool; if True, increases output image
                        size to avoid cropping. Note: horizon
                        line not placed correctly if using this
                        option. TODO: fix this
    Returns:
        im_stab; stabilised image, same shape as input image,
                 im_undist.

    Based on StabilizeImages.m by Michael Schwendeman, 2014.

    NB Results in cropping of the image as output image size 
    is not changed from input image size.
    Sources:
        https://codereview.stackexchange.com/questions/132914/
        crop-black-border-of-image-using-numpy
    """
    # Get image shape for projective transformation
    rows,cols = im_undist.shape

    # Get rotation matrices
    R1 = rotation_matrix(inc=inc1, roll=roll1, azi=azi1)
    R2 = rotation_matrix(inc=inc2, roll=roll2, azi=azi2)
    
    # Combine KR1 and KR2 to make projective matrix R
    KR2 = K @ R2
    KR1 = K @ R1
    # In Matlab: R = (KR2 / KR1)'; right matrix division in numpy:
    R = np.dot(KR2, np.linalg.pinv(KR1))

    if resize_output:
        # Resize output image to avoid cropping
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
            ])
        corners = cv2.perspectiveTransform(np.float32([corners]), R)[0]
        print(corners)
        # Find the bounding rectangle
        bx, by, bwidth, bheight = cv2.boundingRect(corners)

        # Compute the translation homography that will move 
        # (bx, by) to (0, 0)
        T = np.array([
            [ 1, 0, -bx ],
            [ 0, 1, -by ],
            [ 0, 0,   1 ]
            ])
        # Combine warp and translation matrices
        RT = R @ T

        # Perform projective transformation
        im_stab = cv2.warpPerspective(im_undist, RT, (bwidth,bheight))

        # Crop out excessive black borders
        mask = im_stab > 0 # mask of non-black pixels
        # Coordinates of non-black pixels
        coords = np.argwhere(mask)
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1
        # Crop
        im_stab = im_stab[x0:x1, y0:y1]
        # Calculate difference between output and input image sizes
        col_offset = bwidth - cols
        row_offset = bheight - rows
    else:
        im_stab = cv2.warpPerspective(im_undist, R, (cols,rows))

    return im_stab


def reproject_coordinates(projection,
        inc, roll, azi, K, inc2=None, roll2=None, azi2=None, 
        u=None, v=None, H=None, x=None, y=None, z=None,
        ):
    """
    Parameters:
        im_undist - ndarray; image to reproject
        projection - str; one of ['image2world', 'image2image'
                     'world2image', 'world2world']
        inc - float; camera incidence angle
        roll - float; camera roll angle
        azi - float; camera azimuth angle
        K - ndarray; 3x3 upper-triangular camera intrinsic matrix
        inc2 - float; reprojected camera incidence angle
        roll2 - float; reprojected camera roll angle
        azi2 - float; reprojected camera azimuth angle
        u - 2D ndarray or tuple; image u pixel coordinates (columns)
        v - 2D ndarray or tuple; image v pixel coordinates (rows)
        H - float; camera height (in m)
        x - 2D ndarray; world x coordinates 
        y - 2D ndarray; world y coordinates 
        z - 2D ndarray; world z coordinates 
    Returns:
        Reprojected coordinates x,y or u,v depending on
        requested projection
    """
    # Check if inputs are tuple or ndarray
    if u is not None:
        if type(u) is tuple:
            u_range = np.arange(u[0], u[1], dtype=int)
            v_range = np.arange(v[0], v[1], dtype=int)
            # Make 2D arrays
            u, v = np.meshgrid(u_range, v_range)

    if projection == 'image2world':
        if u is None or v is None or H is None:
            raise ValueError('Must input u, v and H')
        # Make 3D rotation matrix
        R = rotation_matrix(inc=inc, roll=roll, azi=azi)
        # Generate transformation matrix P for perspective
        # equation (Eq. 2 in Schwendeman & Thomson 2015)
        P1 = projection_matrix(K)
        P2 = projection_matrix(R)
        P = P1 @ P2
        # Generate matrix from pixel coordinates
        UV =  pixel_matrix(u, v)
        # Do left matrix division: pw = P \ UV (Matlab notation)
        pw = np.linalg.lstsq(P,UV)[0]
        # Transform to earth coordinates
        x = (-pw[0,:] / pw[2,:] * H).reshape(u.shape)
        y = (-pw[1,:] / pw[2,:] * H).reshape(u.shape)
        return x, y

    elif projection == 'image2image':
        if u is None or v is None:
            raise ValueError('Must input u, v')
        if inc2 is None or roll2 is None or azi2 is None:
            raise ValueError('Must input inc2, roll2, azi2')
        # Make 3D rotation matrices
        R1 = rotation_matrix(inc=inc, roll=roll, azi=azi)
        R2 = rotation_matrix(inc=inc2, roll=roll2, azi=azi2)
        # Make projection matrices
        PK = projection_matrix(K)
        PR2 = projection_matrix(R2)
        P2 = PK @ PR2
        PR1 = projection_matrix(R1)
        P1 = PK @ PR1
        # Generate matrix from pixel coordinates
        UV =  pixel_matrix(u, v)
        # Left matrix division uv1 = P1 \ UV:
        uv1 = np.linalg.lstsq(P1,UV)[0]
        # Multiply P2 by uv1
        uv2 = P2 @ uv1
        # Transform  to output pixel coordinates
        v2 = (uv2[1,:] / uv2[2,:]).reshape(u.shape)
        u2 = (uv2[0,:] / uv2[2,:]).reshape(u.shape)
        return u2, v2

    elif projection == 'world2image':
        if x is None or y is None or H is None:
            raise ValueError('Must input x, y and H')
        # Generate required matrices
        R = rotation_matrix(inc=inc, roll=roll, azi=azi)
        PK = projection_matrix(K)
        PR = projection_matrix(R2)
        XY = xy_matrix(x, y, H)
        # Multiply matrices
        xs = PK @ PR @ XY
        # Get pixel coordinates
        u = (xs[0,:] / xs[2,:]).reshape(x.shape)
        v = (xs[1,:] / xs[2,:]).reshape(x.shape)
        return u, v

    elif projection == 'world2world':
        if x is None or y is None or z is None:
            raise ValueError('Must input x, y and z')
        # Rotation matrix
        R = rotation_matrix(inc=inc, roll=roll, azi=azi)
        # Generate XYZ matrix
        XYZ = np.zeros((4, x.flatten().shape[0]))
        XYZ[0,:] = x.flatten()
        XYZ[1,:] = y.flatten()
        XYZ[2,:] = z.flatten()
        XYZ[3,:] = np.ones(x.flatten().shape[0])
        # Left matrix division
        pw = np.linalg.lstsq(R,XYZ)[0]
        x2 = (pw[0,:] / pw[3,:]).reshape(x.shape)
        y2 = (pw[1,:] / pw[3,:]).reshape(x.shape)
        z2 = (pw[2,:] / pw[3,:]).reshape(x.shape)
        return x2, y2, z2

    else:
        raise ValueError(('projection must be one of ' 
                '["image2world", "image2image" '
                '"world2image", "world2world"]'))


class MotionCorrection():
    """
    Main image stabilisation class. Uses the horizon-detection
    method of Schwendeman and Thomson (2015).         

    Init parameters:
        K - ndarray; 3x3 upper-triangular camera intrinsic matrix
        dist - ndarray; lens distortion coefficients
        H - float; camera height (in m) over m.s.l.
        incStab - float; stabilised camera inclination angle
        rollStab - float; stabilised roll angle
        aziStab - float; stabilised azimuth angle

    Citation: Schwendeman, M., J. Thomson, 2015: 
              "A Horizon-tracking Method for Shipboard Video
              Stabilization and Rectification."  
              J. Atmos. Ocean. Tech.
    """
    def __init__(self, K, dist, 
            H=16.4, 
            incStab=72*np.pi/180, 
            rollStab=0*np.pi/180, 
            aziStab=0*np.pi/180,
            ):
        self.K = K # 3x3 upper-triangular camera intrinsic matrix
        self.dist = dist # lens distortion coefficients
        self.H = H # camera height (in m) over m.s.l.
        # Input stabilized camera pose
        self.incStab = incStab; # stabilised inclination angle
        self.rollStab = rollStab; # stabilised roll angle
        self.aziStab = aziStab; # stabilised azimuth angle


    def stabilise_horizon(self, im, ROI_hor, azi, r_hor=None,
            theta_hor=None, return_orientation=True):
        """
        Stabilise input image, im, using parameters defined when
        initialising HorizonStabilisation object.

        Parameters:
            im - ndarray; (grayscale) image array to stabilise
            azi - float; ship heading
            ROI_hor - ndarray; ROI for horizon detection
            r_hor - horizon line r value; if None, will find using
                    find_horizon()
            theta_hor - horizon line theta value (radians)
            return_orientation - bool; if True, return edges, r_max,
                                 theta_max, inc, roll in addition to
                                 im_stab and im_undist

        Returns: 
            im_stab - ndarray; stabilised image
            im_undist - ndarray; undistorted image
        """

        # Undistort image using OpenCV
        print('Undistorting image ... \n')
        im_undist = cv2.undistort(im, cameraMatrix=self.K,
                distCoeffs=self.dist)

        # Crop out casing edges for horizon detection using ROI_hor
        im_undist = im_undist[ROI_hor[0]:ROI_hor[2],
                ROI_hor[1]:ROI_hor[3]]

        # Undistorted image shape (nv, nu) = (rows, columns)
        nv, nu = im_undist.shape[:2]

        # Find horizon pixels in stabilised image
        r_stab, theta_stab = angles_to_horizon(inc=self.incStab,
                roll=self.rollStab, K=self.K)
        u_hor = np.array([0, nu-1]) # Use all columns (u) for horizon
        v_hor_stab = horizon_pixels(r=r_stab, theta=theta_stab,
                u=u_hor)

        # Find horizon
        print('Finding horizon ... \n')
        r_max, theta_max, edges = find_horizon(im_undist)

        # Find camera angles and stabilise images
        inc, roll = horizon_to_angles(r=r_max, theta=theta_max,
                K=self.K)
        print('Stabilising image ... \n')
        im_stab = stabilise_image(im_undist, inc1=inc,
                roll1=roll, azi1=azi, inc2=self.incStab, 
                roll2=self.rollStab, azi2=self.aziStab,
                K=self.K)

        if return_orientation:
            return (im_stab, im_undist, edges, r_max, theta_max, inc,
                    roll)
        else:
            return im_stab, im_undist


    def image_to_world(self, u, v, inc, roll, azi):
        """
        Calls reproject_coordinates().
        """
        x, y = reproject_coordinates(projection='image2world',
                inc=inc, roll=roll, azi=azi, u=u, v=v, K=self.K,
                H=self.H)
        return x, y

    def image_to_image(self, u, v, inc, roll, azi):
        """
        Calls reproject_coordinates().
        """
        u2, v2 = reproject_coordinates(projection='image2image',
                inc=inc, roll=roll, azi=azi, inc2=self.incStab,
                roll2=self.rollStab, azi2=self.aziStab, K=self.K,
                H=self.H, u=u, v=v)
        return u2, v2


    def stabilise_IMU():
        """
        Stabilise image using IMU data.
        """
        return True



if __name__ == '__main__':
    """
    Test/example script.
    """
    from argparse import ArgumentParser
    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-datadir", 
                help=("Path to data_root (work) directory"),
                type=str,
                default='/home/mikapm/stereo-wave/WhiteCapFraction',
                )
        return parser.parse_args(**kwargs)
    args = parse_args(args=sys.argv[1:])

    # Open test image from whitecapping directory
    datadir = '/home/mikapm/stereo-wave/WhiteCapFraction'
    images = sorted(glob.glob(os.path.join(datadir, 'Images',
        '20191210_1817UTC', 'flea83_*.pgm')))
    # Read in one image to get shape
    im = cv2.imread(images[0], 0) # flag=0 reads image in grayscale
    nv, nu = im.shape[:2]
    # ROI_hor = [v_min, u_min, v_max, u_max]
    ROI_hor = [0, 120, nv, nu-50] # for cropping out casing edges

    # Read intrinsic parameters from correct .mat file
    fn0 = images[0].split('/')[-1]
    if fn0[4:6] == '83':
        fn_intr = 'intr_STBD.mat'
    elif fn0[4:6] == '65':
        fn_intr = 'intr_PORT.mat'
    fn_intr = os.path.join(datadir, 'CalibrationFiles', fn_intr)
    cameraParams = spio.loadmat(fn_intr)
    K = cameraParams['K'].T # intrinsic matrix
    dist = np.zeros(5) # radial distortion params vector
    dist[0:2] = cameraParams['dist'].squeeze() # only have k1,k2

    # Initialise MotionCorrection object
    MC = MotionCorrection(K=K, dist=dist)
    # Stabilise and reproject images
    images = [images[20]]
    for i,fn in enumerate(images):
        print(' %s \n' % i)
        im = cv2.imread(os.path.join(datadir, 'Images', fn), 0)
        # Stabilise
        im_stab, im_undist, edges, r_max, theta_max, inc, roll = MC.stabilise_horizon(
                im, ROI_hor, 0)
        # Sample reprojection coordinates
        # TODO: make into input args
        nv_u, nu_u = im_undist.shape[:2]
        u_samp = np.arange(nu_u/2-200, nu_u/2+200, dtype=int)
        v_samp = np.arange(400, 650, dtype=int)
        u_box, v_box = np.meshgrid(u_samp, v_samp)
        x_samp = np.arange(-30, 30, dtype=int)
        y_samp = np.arange(40, 100, dtype=int)
        # Reproject square
        x_box, y_box = MC.image_to_world(u=u_box, v=v_box,
                inc=inc, roll=roll, azi=0)
        u_box_stab, v_box_stab = MC.image_to_image(u=u_box,
                v=v_box, inc=inc, roll=roll, azi=0)

        # Test plot horizon line
        u_hor = np.array([0,im_undist.shape[1]-1])
        v_hor = horizon_pixels(r=r_max, theta=theta_max, u=u_hor)
        fig, ax = plt.subplots(1, 3, figsize=(20,6))
        # Plot horizon in undistorted image
        ax[0].imshow(im_undist, cmap=plt.cm.gray, aspect='auto')
        # Plot edges of box
        ax[0].plot(u_hor, v_hor, color='r', linestyle='--')
        ax[0].plot(u_box[:,1],v_box[:,1],'-g')
        ax[0].plot(u_box[:,-1],v_box[:,-1],'-g')
        ax[0].plot(u_box[1,:],v_box[1,:],'-g')
        ax[0].plot(u_box[-1,:],v_box[-1,:],'-g')
        ax[0].set_title('Undistorted image', fontsize=16)
        # Plot stabilised image
        ax[1].imshow(im_stab, cmap=plt.cm.gray, aspect='auto')
        ax[1].plot(u_box_stab[:,1],v_box_stab[:,1],'-g')
        ax[1].plot(u_box_stab[:,-1],v_box_stab[:,-1],'-g')
        ax[1].plot(u_box_stab[1,:],v_box_stab[1,:],'-g')
        ax[1].plot(u_box_stab[-1,:],v_box_stab[-1,:],'-g')
        ax[1].set_title('Stabilised image', fontsize=16)
        # Plot reprojected image
        ax[2].pcolormesh(x_box, y_box, 
                im_undist[v_samp[0]:v_samp[-1]+1,u_samp[0]:u_samp[-1]+1],
                cmap=plt.cm.gray, vmin=0, vmax=255)
        # Plot edges of box
        ax[2].plot(x_box[:,1],y_box[:,1],'-g')
        ax[2].plot(x_box[:,-1],y_box[:,-1],'-g')
        ax[2].plot(x_box[1,:],y_box[1,:],'-g')
        ax[2].plot(x_box[-1,:],y_box[-1,:],'-g')
        # Set axis limits
        ax[2].set_xlim(x_samp[0]-10, x_samp[-1]+10)
        ax[2].set_ylim(y_samp[0]-10, y_samp[-1]+10)
        # Set background color to black
        ax[2].set_facecolor((0.0, 0.0, 0.0))
        ax[2].set_aspect('auto')
        ax[2].set_title('Reprojected image', fontsize=16)
        #plt.savefig(os.path.join(
        #    datadir, 'Figures', 'py_stab', 'py_stab_%s' % i))
        plt.show()
        #plt.close()





