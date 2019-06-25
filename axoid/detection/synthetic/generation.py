#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to generate the synthetic dataset for the training.

/!\ There are a lot of hard coded numbers. These were empirically tuned so that
the synthetic data looks like real data.

Created on Thu Nov 22 10:01:56 2018

@author: nicolas
"""

import os, pickle
import warnings
import math

import numpy as np
import matplotlib.pyplot as plt
from skimage import draw, exposure, measure
import skimage.morphology as morph
from scipy.stats import multivariate_normal
from scipy import ndimage as ndi
from scipy import signal

from axoid.utils.processing import flood_fill


# Following are pre-computed on real data (corresponding to low laser gain). See stats_###.pkl & README.md.
_BKG_MEAN_R = 0.04061809239988313 # mean value of red background (190222)
_BKG_MEAN_G = 0.03090710807146899 # mean value of red background (190222)
_BKG_STD = 0.005 # Standard deviation of mean value of background (empirically tuned)
_ROI_MAX_1 = 0.2276730082246407 # fraction of red ROI with 1 as max intensity (181121)
_ROI_MAX_MEAN = 0.6625502112855037 # mean of red ROI max (excluding 1.0) (181121)
_ROI_MAX_STD = 0.13925117610178622 # std of red ROI max (excluding 1.0) (181121)

# Threshold for the laser gain where saturation starts to occur
_GAIN_T = 0.5

# Following are the pre-generated GCaMP response kernel
with open(os.path.join(os.path.dirname(__file__), "GCaMP_kernel.pkl"), "rb") as f:
    kernel_f, kernel_s = pickle.load(f)


def create_neurons(n_neurons, shape, return_label):
    """
    Return gaussian and segmentation images corresponding to neurons.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons to be present on the stack.
    shape : tuple of int
        Tuple (height, width) representing the shape of the images.
    return_label : bool (default = False)
        If True, synth_seg will be the labels of the neurons instead of just
        their binary segmentations.
    
    Returns
    -------
    gaussians : ndarray
        Stack of images, where one image has one gaussian corresponding to one
        neuron.
    neuron_segs : ndarray
        Stack of binary images, which corresponds to the detection of these
        neurons. Will be used as ground truths for deep learning.
    """
    ellipse_size = 1.5 # factor for the ground truth ellipse (normalized by std)
    # Meshgrid for the gaussian weights
    rows, cols = np.arange(shape[0]), np.arange(shape[1])
    meshgrid = np.zeros(shape + (2,))
    meshgrid[:,:,0], meshgrid[:,:,1] = np.meshgrid(cols, rows) # note the order
    
    gaussians = np.zeros((n_neurons,) + shape)
    if return_label:
        neuron_segs = np.zeros((n_neurons,) + shape, dtype=np.uint8)
    else:
        neuron_segs = np.zeros((n_neurons,) + shape, dtype=np.bool)
    for i in range(n_neurons):
        # Loop until the randomly generated neuron is in the image 
        # and doesn't overlap with another (can theoretically loop to infinity if too many neurons)
        while True:
            # Mean and covariance matrix of gaussian (empirically tuned)
            # Note that x and y axes correspond to col and row 
            mean = np.array([np.random.randint(shape[1]), np.random.randint(shape[0])])
            scale_x = shape[1] / 64
            scale_y = shape[0] / 64
            cross_corr = np.random.randint(-2, 3) * min(scale_x, scale_y)
            cov = np.array([[np.random.randint(1, 4) * scale_x, cross_corr],
                            [cross_corr, np.random.randint(10, 40) * scale_y]])

            # Bounding ellipse
            val, vec = np.linalg.eig(cov)
            rotation = math.atan2(vec[0, np.argmax(val)], vec[1, np.argmax(val)])
            rr, cc = draw.ellipse(mean[1], mean[0], 
                                  ellipse_size * np.sqrt(val[1]), 
                                  ellipse_size * np.sqrt(val[0]),
                                  rotation=rotation)
            # Check if outside the image
            if (rr < 0).any() or (rr >= shape[0]).any() or (cc < 0).any() or (cc >= shape[1]).any():
                continue
            else:
                # Check if overlapping/touching with any existing neuron
                tmp_mask = np.zeros(shape, dtype=np.bool)
                tmp_mask[rr,cc] = 1
                if (neuron_segs[i, morph.dilation(tmp_mask)] != 0).any():
                    continue
                else:
                    break
        neuron_segs[i, rr, cc] = 1 + i * return_label
        
        # Create gaussian weight image
        gaussians[i,:,:] = multivariate_normal.pdf(meshgrid, mean, cov)
        gaussians[i,:,:] /= gaussians[i,:,:].sum()
    
    return gaussians, neuron_segs


def get_flurophores(n_neurons, n_images, gcamp_type):
    """
    Return the tdTomato level, and GCaMP dynamics.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons to be present on the stack.
    n_images : int
        Number of images in the stack.
    gcamp_type : str
        Type of GCaMP fluorophore to simulate, "6s" or "6f".
    
    Returns
    -------
    tdTom_max : ndarray
        Array of the maximum intensity of tdTomato for each neuron.
    gcamp_dynamics : ndarray
        Array of the maximum intensity of GCaMP for each neuron and each frame.
        This is to simulate dynamics of neural activations.
    """
    fps = 2.4 # synthetic frame per seconds
    # Choose which fluorophores are expressed for each neuron
    c_presence = np.array([[True, True], [True, False], [False, True]], dtype=np.bool)
    channel_neurons = c_presence[np.random.choice(len(c_presence), size=n_neurons, p=[0.9, 0.05, 0.05]), :]
    
    # tdTomato: choose max intensity of the neurons
    tdTom_max = np.zeros(n_neurons)  
    for i in range(n_neurons):
        if channel_neurons[i, 0] == False:
            continue
        # Sample randomly the neuron maximum 
        if np.random.rand() < _ROI_MAX_1:
            tdTom_max[i] = 1.0
        else:
            loc = _ROI_MAX_MEAN
            scale = _ROI_MAX_STD
            tdTom_max[i] = np.clip(np.random.normal(loc=loc, scale=scale), 0, 1)
    
    # GCaMP: create dynamics through time of the neurons
    gcamp_dynamics = np.zeros((n_neurons, n_images))
    # GCaMP type (50% 6f and 50% 6s)
    if gcamp_type == "6f": # GCaMP6f
        kernel_gcamp = kernel_f
    elif gcamp_type == "6s": # GCaMP6s
        kernel_gcamp = kernel_s
    else:
        raise ValueError("Unknown GCaMP type '{}'.".format(gcamp_type))
    t = np.arange(np.ceil(n_images / fps) * 1000) / 1000 # timesteps in ms
    for i in range(n_neurons):
        if channel_neurons[i, 1] == False:
            continue
        # Rate of firing
        rate = np.zeros(len(t))
        rate[np.random.randint(len(t), size=n_images // 10)] = 0.5
        rate = np.convolve(rate, signal.gaussian(5000, 1000), mode='full')[:len(rate)].clip(0,1)
        # Spiking (80%) or non-spiking (20%)
        if np.random.rand() < 0.8:
            spikes = np.random.poisson(rate / 250, size=len(t)).clip(0,1)
            dynamics = np.convolve(spikes, kernel_gcamp, mode="full")[:len(spikes)]
        else:
            dynamics = np.convolve(rate / 100, kernel_gcamp, mode="full")[:len(rate)]
        # Sub-sample to fps
        gcamp_dynamics[i] = dynamics[::int(np.rint(1000/fps))][:n_images]
        # If no red, assures a minimum to avoid invisible neurons
        if channel_neurons[i,0]:
            gcamp_dynamics[i] = gcamp_dynamics[i].clip(0,1)
        else:
            gcamp_dynamics[i] = gcamp_dynamics[i].clip(np.random.normal(0.4, 0.02),1)
            
    return tdTom_max, gcamp_dynamics


def deform_neurons(n_neurons, shape, gaussians, neuron_segs):
    """
    Deform neurons and segmentation like the real acquisition system.
    
    Parameters
    ----------
    n_neurons : int
        Number of neurons to be present on the stack.
    shape : tuple of int
        Tuple (height, width) representing the shape of the images.
    gaussians : ndarray
        Stack of images, where one image has one gaussian corresponding to one
        neuron.
    neuron_segs : ndarray
        Stack of binary images, which corresponds to the detection of these
        neurons. Will be used as ground truths for deep learning.
    
    Returns
    -------
    wrp_gaussian : ndarray
        Like gaussian, except they have been deformed.
    wrp_seg : ndarray
        Binary images corresponding to wrp_gaussian detections.
    """
    # For warping (like acquisition process):
    k_s = 50 # size of kernel for smoothing translations (in number of rows)
    n_r = 0.5 # number of rows after which the standard deviation of the translations are 1
    # Smoothing kernel for the translations
    kernel = signal.gaussian(k_s * shape[1], k_s * shape[1] / 2 ** (5/2))
    kernel /= kernel.sum()
    
    # Create horizontal and vertical translations
    trans_row = np.cumsum(np.random.normal(0, 1 / np.sqrt(n_r * shape[1]), size=shape[0] * shape[1]))
    trans_col = np.cumsum(np.random.normal(0, 1 / np.sqrt(n_r * shape[1]), size=shape[0] * shape[1]))
    trans_row = np.rint(np.convolve(trans_row, kernel, mode="same").reshape(shape))
    trans_col = np.rint(np.convolve(trans_col, kernel, mode="same").reshape(shape))

    # Warp gaussians and segmentations defining neurons
    wrp_gaussian = np.zeros_like(gaussians)
    wrp_seg = np.zeros_like(neuron_segs)
    for r in range(wrp_gaussian.shape[1]):
        for c in range(wrp_gaussian.shape[2]):        
            trans_r = int(r + trans_row[r,c])
            trans_c = int(c + trans_col[r,c])

            # Sample if inside the image, else will be 0s
            if 0 < trans_r and trans_r < wrp_gaussian.shape[1] - 1 and \
               0 < trans_c and trans_c < wrp_gaussian.shape[2] - 1:
                wrp_gaussian[:, r, c] = gaussians[:, trans_r, trans_c]
                wrp_seg[:, r, c] = neuron_segs[:, trans_r, trans_c]
    
    for i in range(n_neurons):
        # Normalize the gaussians
        wrp_gaussian[i] /= wrp_gaussian[i].sum()
        # If warping the segmentation created multiple ROIs, only keep the largest one
        labels, num = measure.label(wrp_seg[i], connectivity=1, return_num=True)
        if num > 1:
            regions = measure.regionprops(labels)
            areas = [region.area for region in regions]
            largest_label = regions[np.argmax(areas)].label
            wrp_seg[i][labels != largest_label] = 0
        # Fill the possible holes in the warped segmentation
        wrp_seg[i] = flood_fill(np.pad(wrp_seg[i], 1, 'constant'), fill_val=wrp_seg[i].max())[1:-1, 1:-1]
            
    return wrp_gaussian, wrp_seg


def sample_neurons(i, n_neurons, shape, n_samples, wrp_gaussian, wrp_seg,
                   tdTom_max, gcamp_dynamics, laser_gain):
    """
    Sample from the gaussians defining neurons for frame `i`.
    
    Parameters
    ----------
    i : int
        Time index of the frame which is being sampled.
    n_neurons : int
        Number of neurons to be present on the stack.
    shape : tuple of int
        Tuple (height, width) representing the shape of the images.
    n_samples : int
        Parameter over the number of samples to draw to get pixelated neurons.
    wrp_gaussian : ndarray
        Like gaussian, except they have been deformed.
    wrp_seg : ndarray
        Binary images corresponding to wrp_gaussian detections.
    tdTom_max : ndarray
        Array of the maximum intensity of tdTomato for each neuron.
    gcamp_dynamics : ndarray
        Array of the maximum intensity of GCaMP for each neuron and each frame.
        This is to simulate dynamics of neural activations.
    laser_gain : float
        Fake laser gain, from 0 to 1 (min to max).
    
    Returns
    -------
    wrp_neuron : ndarray
        RGB image corresponding to synthetic fluorophore signals from the neurons.
    """
    wrp_neuron = np.zeros(shape + (3,), dtype=wrp_gaussian.dtype)
    for j in range(n_neurons):
        # Only sample if neuron is in image
        if wrp_seg[j].max() == 0:
            continue
        for c in [0,1]:
            if c == 0:
                max_neuron = tdTom_max[j]
            elif c == 1:
                max_neuron = gcamp_dynamics[j,i]
            # Only if channel is not 0 (with a tolerance)
            if max_neuron < 1e-8: 
                continue
            # Sample from gaussians
            # Sampling is adjusted for neuron's intensity, size, and laser gain
            if laser_gain < _GAIN_T or c == 0: # low gain or tdTomato, no reduced sampling by laser gain
                x = np.random.choice(np.arange(shape[0] * shape[1]), 
                                     size=int(n_samples[j,c] * \
                                              max_neuron ** 0.5 * (np.count_nonzero(wrp_seg[j]) / 150)), 
                                     p=wrp_gaussian[j].ravel())
            else: # high gain, reduced sampling
                x = np.random.choice(np.arange(shape[0] * shape[1]), 
                                     size=int(n_samples[j,c] * (1 - 0.5 * (laser_gain - _GAIN_T) / (1 - _GAIN_T)) * \
                                              max_neuron ** 0.5 * (np.count_nonzero(wrp_seg[j]) / 150)), 
                                     p=wrp_gaussian[j].ravel())
            # If no sampling (area too small, low intensity, and high gain for e.g.), continue 
            if x.size == 0:
                continue
            y, x = np.unravel_index(x, shape)
            hist = plt.hist2d(x, y, bins=[shape[1], shape[0]], range=[[0, shape[1]], [0, shape[0]]])[0]
            plt.close()

            # Adjust maximum and saturation
            hist = hist.T / hist.max() * (max_neuron + (1 - max_neuron) * laser_gain / _GAIN_T)
            if laser_gain >= _GAIN_T: # high gain, saturation occuring
                hist = hist.clip(0, np.percentile(hist[hist > 0], 
                                                  100 * (1 - (laser_gain - _GAIN_T) / (1 - _GAIN_T)) + \
                                                  80 * (laser_gain - _GAIN_T) / (1 - _GAIN_T)))
                hist /= hist.max()
            hist = hist.clip(0,1)

            wrp_neuron[...,c] = np.maximum(wrp_neuron[...,c], hist)
    
    return wrp_neuron


def reduce_with_border(wrp_seg, return_label):
    """
    Reduce the segmentation to one image, while making sure ROIs are separated.
    
    Parameters
    ----------
    wrp_seg : ndarray
        Stack of binary images corresponding to their ROI, or label.
    return_label : bool (default = False)
        If True, synth_seg will be the labels of the neurons instead of just
        their binary segmentations.
    
    Returns
    -------
    wrp_seg : ndarray
        Single binary images corresponding the to projection of input wrp_seg,
        but with potential separation border (with background pixels) between
        ROI that are touching.
    """
    # Compute number of neurons inside the frame
    n_neurons = 0
    for j in range(wrp_seg.shape[0]):
        if np.count_nonzero(wrp_seg[j]) > 0:
            n_neurons += 1
    # Make sure that touching warped neurons are not segmented together
    # and reduce the segmentation to one image per frame
    _, num = measure.label(wrp_seg.max(0).astype(np.bool), return_num=True, connectivity=1)
    if num < n_neurons:
        # If so, introduce a background border with watershed
        dist = []
        dist_label = []
        for j in range(wrp_seg.shape[0]):
            if np.count_nonzero(wrp_seg[j]) == 0:
                continue
            dist.append(ndi.distance_transform_edt(wrp_seg[j]))
            dist_label.append(wrp_seg[j].max())
        dist = np.array(dist)

        local_maxi = np.zeros(dist[0].shape, dtype=np.uint8)
        for j in range(dist.shape[0]):
            r,c = np.unravel_index(np.argmax(dist[j], axis=None), dist[j].shape)
            local_maxi[r,c] = dist_label[j]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = morph.watershed(-dist.max(0), local_maxi, mask=wrp_seg.max(0).astype(np.bool), 
                                     watershed_line=True)
        if return_label:
            wrp_seg = labels
        else:
            wrp_seg = labels.astype(np.bool)
    else:
        wrp_seg = wrp_seg.max(0)
    
    return wrp_seg


def warp_neurons(n_images, n_neurons, shape, gaussians, neuron_segs, return_label,
                 tdTom_max, gcamp_dynamics, laser_gain, cyan_gcamp):
    """
    Return the warped and sampled neurons and segmentations.
    
    Parameters
    ----------
    n_images : int
        Number of images in the stack.
    n_neurons : int
        Number of neurons to be present on the stack.
    shape : tuple of int
        Tuple (height, width) representing the shape of the images.
    gaussians : ndarray
        Stack of images, where one image has one gaussian corresponding to one
        neuron.
    neuron_segs : ndarray
        Stack of binary images, which corresponds to the detection of these
        neurons. Will be used as ground truths for deep learning.
    return_label : bool (default = False)
        If True, synth_seg will be the labels of the neurons instead of just
        their binary segmentations.
    tdTom_max : ndarray
        Array of the maximum intensity of tdTomato for each neuron.
    gcamp_dynamics : ndarray
        Array of the maximum intensity of GCaMP for each neuron and each frame.
        This is to simulate dynamics of neural activations.
    laser_gain : float
        Fake laser gain, from 0 to 1 (min to max).
    cyan_gcamp : bool (default = False)
        If True, the GCaMP will appear cyan (same in green and blue channel).
        Else, it will be green (0 in blue channel).
    
    Returns
    -------
    wrp_neurons : ndarray
        Stack of RGB images of synthetic tdTomato and GCaMP signals of warped
        neurons.
    wrp_segs : ndarray
        Stack of binary images corresponding to their ROI, or label.
    """
    # Number of samples for each neuron (empirically tuned)
    n_samples = np.random.normal(loc=1000, scale=200, size=n_neurons * 2).reshape([-1, 2])
    n_samples = (n_samples + 0.5).astype(np.uint16)
    
    wrp_segs = np.zeros((n_images,) + shape, dtype=neuron_segs.dtype)
    wrp_neurons = np.zeros((n_images,) + shape + (3,), dtype=gaussians.dtype)
    for i in range(n_images):
        # Deform gaussians and segmentations
        wrp_gaussian, wrp_seg = deform_neurons(n_neurons, shape, gaussians, neuron_segs)
        
        # Sample neurons
        wrp_neurons[i] = sample_neurons(i, n_neurons, shape, n_samples, wrp_gaussian, wrp_seg,
                                        tdTom_max, gcamp_dynamics, laser_gain)
        
        # Reduce segmentations to one image
        wrp_segs[i] = reduce_with_border(wrp_seg, return_label)
        
    # Make GCaMP cyan if applicable
    if cyan_gcamp:
        wrp_neurons[...,2] = wrp_neurons[...,1]
        
    return wrp_neurons, wrp_segs


def create_noise(n_images, shape, laser_gain, cyan_gcamp):
    """
    Create noisy background for all frames.
    
    Parameters
    ----------
    n_images : int
        Number of images in the stack.
    shape : tuple of int
        Tuple (height, width) representing the shape of the images.
    laser_gain : float
        Fake laser gain, from 0 to 1 (min to max).
    cyan_gcamp : bool (default = False)
        If True, the GCaMP will appear cyan (same in green and blue channel).
        Else, it will be green (0 in blue channel).
    
    Returns
    -------
    noise : ndarray
        Stack of RGB images with only the noise.
    """
    if laser_gain < _GAIN_T: # low gain, no saturation
        noise_means = np.array([_BKG_MEAN_R * (1 - laser_gain) + 0.14 * laser_gain,
                                _BKG_MEAN_G * (1 - laser_gain) + 0.13 * laser_gain])
        noise_means = np.array([np.random.normal(loc=noise_means[0], scale=_BKG_STD),
                                np.random.normal(loc=noise_means[1], scale=_BKG_STD)])

        noise_tdTom = np.random.exponential(noise_means[0], size=(n_images,) + shape)
        noise_gcamp = (np.random.binomial(1, 1 + (0.03 - 1) * laser_gain ** 0.5, size=(n_images,) + shape) * \
                       np.random.exponential(noise_means[1], size=(n_images,) + shape))

    else: # high gain, saturation and reduced sampling
        noise_means = np.array([_BKG_MEAN_R * (1 - laser_gain) + 0.14 * laser_gain,
                                (_BKG_MEAN_G * (1 - _GAIN_T) + 0.13 * _GAIN_T) * \
                                (1 - (laser_gain - _GAIN_T) ** 2 / (1 - _GAIN_T) ** 2) + \
                                0.4 * (laser_gain - _GAIN_T) ** 2 / (1 - _GAIN_T) ** 2])
        noise_means = np.array([np.random.normal(loc=noise_means[0], scale=_BKG_STD),
                                np.random.normal(loc=noise_means[1], scale=_BKG_STD)])

        noise_tdTom = np.maximum(
            np.random.exponential(noise_means[0], size=(n_images,) + shape),
            np.random.binomial(1, 0.004 * (laser_gain - _GAIN_T) / (1 - _GAIN_T), size=(n_images,) + shape))
        noise_gcamp = np.maximum(
            np.random.binomial(1, 1 + (0.03 - 1) * laser_gain ** 0.5, size=(n_images,) + shape) * \
            np.random.exponential(noise_means[1], size=(n_images,) + shape),
            np.random.binomial(1, 0.005 * (laser_gain - _GAIN_T) / (1 - _GAIN_T), size=(n_images,) + shape))
    
    # Make GCaMP cyan if applicable
    if cyan_gcamp:
        noise = np.stack([noise_tdTom, noise_gcamp, noise_gcamp], -1).clip(0,1)
    else:
        noise = np.stack([noise_tdTom, noise_gcamp, np.zeros_like(noise_gcamp)], -1).clip(0,1)
    
    return noise

## This is the main function
def synthetic_stack(shape, n_images, n_neurons, cyan_gcamp=False, return_label=False):
    """
    Return a stack of synthetic neural images.
    
    Parameters
    ----------
    shape : tuple of int
        Tuple (height, width) representing the shape of the images.
    n_images : int
        Number of images in the stack.
    n_neurons : int
        Number of neurons to be present on the stack.
    cyan_gcamp : bool (default = False)
        If True, the GCaMP will appear cyan (same in green and blue channel).
        Else, it will be green (0 in blue channel).
    return_label : bool (default = False)
        If True, synth_seg will be the labels of the neurons instead of just
        their binary segmentations.
            
    Returns
    -------
    synth_stack : ndarray
        Stack of N synthetic images.
    synth_seg : ndarray
        Stack of N synthetic segmentations (or labels, see `return_label`).
    """ 
    ## Initialization
    # GCaMP type (50% 6f and 50% 6s)
    if np.random.rand() < 0.5:
        gcamp_type = "6f"
    else: # GCaMP6s
        gcamp_type = "6s"
    # Laser gain, from 0 to 1 <=> low to high
    laser_gain = 1 - 0.5 * signal.gaussian(100, 20)
    laser_gain = np.random.choice(np.arange(100) / 100, p=laser_gain / laser_gain.sum())
    
    # Create the gaussians representing the neurons
    gaussians, neuron_segs = create_neurons(n_neurons, shape, return_label)
    
    # Choose which channels are present in each neurons
    tdTom_max, gcamp_dynamics = get_flurophores(n_neurons, n_images, gcamp_type)
    
    # Warp neurons for each image to create the stack
    wrp_neurons, wrp_segs = warp_neurons(n_images, n_neurons, shape, gaussians, neuron_segs, return_label,
                                         tdTom_max, gcamp_dynamics, laser_gain, cyan_gcamp)
                    
    # Add background noise
    noise = create_noise(n_images, shape, laser_gain, cyan_gcamp)
    
    # Put neurons and noise together
    synth_stack = np.maximum(wrp_neurons, noise)
    for c in [0,1,2]:
        if c == 2 and cyan_gcamp is False:
            continue
        synth_stack[...,c] /= synth_stack[...,c].max()
    synth_seg = wrp_segs
    
    # Random gamma correction (image should be in [0,1] range)
    gamma = np.random.rand() * 0.6 + 0.7 # in [0.7, 1.3)
    synth_stack = exposure.adjust_gamma(synth_stack, gamma=gamma)
    
    return synth_stack, synth_seg
