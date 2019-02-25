#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate the synthetic dataset for the training.
See synthetic_generation.ipynb for more details.
Created on Thu Nov 22 10:01:56 2018

@author: nicolas
"""

import os, time
import warnings
import math

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, draw, exposure
from scipy.stats import multivariate_normal
from imgaug import augmenters as iaa

from utils_common.image import to_npint
from utils_common.processing import flood_fill


# Following are pre-computed on real data. See stats_###.pkl & README.md.
_BKG_MEAN_R = 0.04061809239988313 # mean value of red background (190222)
_BKG_MEAN_G = 0.03090710807146899 # mean value of red background (190222)
_BKG_STD = 0.005 # Standard deviation of mean value of background (empirically tuned)
_ROI_MAX_1 = 0.2276730082246407 # fraction of red ROI with 1 as max intensity (181121)
_ROI_MAX_MEAN = 0.6625502112855037 # mean of red ROI max (excluding 1.0) (181121)
_ROI_MAX_STD = 0.13925117610178622 # std of red ROI max (excluding 1.0) (181121)


def synthetic_stack(shape, n_images, n_neurons):
    """
    Return a stack of synthetic neural images.
    
    Args:
        shape: tuple of int
            Tuple (height, width) representing the shape of the images.
        n_images: int
            Number of images in the stack.
        n_neurons: int
            Number of neurons to be present on the stack.
            
    Returns:
        synth_stack: ndarray of shape NxHxWx3
            Stack of N synthetic images.
        synth_seg: ndarray of shape NxHxW
            Stack of N synthetic segmentations.
    """ 
    # Initialization
    ellipse_size = 1.5 # factor for the ground truth ellipse (normalized by std)
    # Number of samples for each neuron (empirically tuned)
    n_samples = np.random.normal(loc=1000, scale=200, size=n_neurons * 2).reshape([-1, 2])
    n_samples = (n_samples + 0.5).astype(np.uint16)
    grid_size = 8 # for the elastic deformation
    
    ## Create the gaussians representing the neurons
    gaussians = np.zeros((n_neurons,) + shape)
    neurons_seg = np.zeros(shape, dtype=np.bool)
    # Meshgrid for the gaussian weights
    rows, cols = np.arange(shape[0]), np.arange(shape[1])
    meshgrid = np.zeros(shape + (2,))
    meshgrid[:,:,0], meshgrid[:,:,1] = np.meshgrid(cols, rows) # note the order
    for i in range(n_neurons):
        # Loop until the randomly generated neuron is in the image 
        # and doesn't overlap with another (can theoretically loop to infinity)
        while True:
            # Mean and covariance matrix of gaussian (empirically tuned)
            # Note that x and y axes are col and row (so, inversed!)
            mean = np.array([np.random.randint(shape[1]), np.random.randint(shape[0])])
            scale_x = shape[1] / 50
            scale_y = shape[0] / 50
            cross_corr = np.random.randint(-2, 2) * min(scale_x, scale_y)
            cov = np.array([[np.random.randint(1, 3) * scale_x, cross_corr],
                            [cross_corr, np.random.randint(10, 30) * scale_y]])

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
            # Check if overlapping with any existing neuron
            elif (neurons_seg[rr, cc] == True).any():
                continue
            else:
                break
        neurons_seg[rr, cc] = True
        
        # Create gaussian weight image
        gaussians[i,:,:] = multivariate_normal.pdf(meshgrid, mean, cov)
        gaussians[i,:,:] /= gaussians[i,:,:].sum()

    # Choose which channels are present in each neurons
    c_presence = np.array([[True, True], [True, False], [False, True]], dtype=np.bool)
    channel_neurons = c_presence[np.random.choice(len(c_presence), size=n_neurons, p=[0.9, 0.05, 0.05]), :]
    
    # RED: choose max intensity of the neurons
    red_max = np.zeros(n_neurons)  
    for i in range(n_neurons):
        if channel_neurons[i, 0] == False:
            continue
        # Sample randomly the neuron maximum 
        if np.random.rand() < _ROI_MAX_1:
            red_max[i] = 1.0
        else:
            loc = _ROI_MAX_MEAN
            scale = _ROI_MAX_STD
            red_max[i] = np.clip(np.random.normal(loc=loc, scale=scale), 0, 1)
    
    # GREEN: choose dynamics through time of the neurons
    green_dynamics = np.zeros((n_neurons, n_images))
    kernel = np.exp(- (np.arange(25) - 12)**2 /50)
    for i in range(n_neurons):
        if channel_neurons[i, 1] == False:
            continue
        stimuli = np.zeros(n_images)
        peaks = np.random.randint(n_images, size=n_images // 20)
        stimuli[peaks] = 1.0
        green_dynamics[i] = np.convolve(stimuli, kernel, mode="same")[:n_images]
        # If no red, assures a minimum of 0.4 to avoid invisible neurons
        if channel_neurons[i,0]:
            green_dynamics[i] = green_dynamics[i].clip(0,1)
        else:
            green_dynamics[i] = green_dynamics[i].clip(0.4,1)
    
    ## Warp neurons for each image to create the stack
    # Define warping sequence
    wrpseq = iaa.Sequential([
        iaa.PiecewiseAffine(scale=0.025, nb_rows=grid_size, nb_cols=grid_size)
    ])
    wrp_segs = np.zeros((n_images,) + shape, dtype=np.bool)
    wrp_neurons = np.zeros((n_images,) + shape + (3,), dtype=gaussians.dtype)
    for i in range(n_images):
        # Set the warping to deterministic for warping both neurons and segmentation the same way
        seq_det = wrpseq.to_deterministic()
        
        # Warp the neurons, and create image by sampling
        for j in range(n_neurons):
            # Warp gaussian defining it
            wrp_gaussian = seq_det.augment_image(gaussians[j])
            wrp_gaussian /= wrp_gaussian.sum()
            
            for c in [0,1]:
                if channel_neurons[j,c] == False:
                    continue
                # Sample from gaussians
                x = np.random.choice(shape[0] * shape[1], size=n_samples[j,c], p=wrp_gaussian.ravel())
                y, x = np.unravel_index(x, shape)
                hist = plt.hist2d(x, y, bins=[shape[1], shape[0]], range=[[0, shape[1]], [0, shape[0]]])[0]
                plt.close()
                if c == 0:
                    max_neuron = red_max[j]
                elif c == 1:
                    max_neuron = green_dynamics[j,i]
                wrp_neurons[i,...,c] = np.maximum(wrp_neurons[i,...,c], hist.T / hist.max() * max_neuron)
            
        # Warp the segmentation
        wrp_segs[i] = seq_det.augment_image(neurons_seg)
        # Fill the possible holes in the warped segmentation
        wrp_segs[i] = flood_fill(np.pad(wrp_segs[i], 1, 'constant'))[1:-1, 1:-1]
    
    ## Add noise (sampled from an exponential distribution)
    noise_means = np.array([np.random.normal(loc=_BKG_MEAN_R, scale=_BKG_STD),
                            np.random.normal(loc=_BKG_MEAN_G, scale=_BKG_STD)])
    noise = np.stack([np.random.exponential(scale=noise_means[0], size=(n_images,) + shape),
                      np.random.exponential(scale=noise_means[1], size=(n_images,) + shape),
                      np.zeros((n_images,) + shape)], -1)
    
    synth_stack = np.maximum(wrp_neurons, noise)
    for c in [0,1]:
        synth_stack[...,c] /= synth_stack[...,c].max()
    synth_seg = wrp_segs
    
    ## Random gamma correction (with prior rescaling)
    gamma = np.random.rand() * 0.6 + 0.7 # in [0.7, 1.3)
    synth_stack = exposure.adjust_gamma(synth_stack, gamma=gamma)
    
    return synth_stack, synth_seg


if __name__ == "__main__":
    ## Parameters and constants (shape and n_neurons are below)
    n_stacks = 200
    n_images = 200
    
    date = time.strftime("%y%m%d", time.localtime())
    synth_dir = "/data/talabot/pdm/dataset_cv-annotated/synthetic_%s/" % date
    
    start = time.time()
    for i in range(n_stacks):
        ### Random parameters here ##
        # Randomized shape
        if np.random.rand() < 0.5: # square image half of the time
            rand_size = np.random.randint(6, 10 + 1) * 32
            shape = (rand_size, rand_size)
        else:
            rand_h = np.random.randint(6, 10 + 1) * 32
            rand_w = np.random.randint(rand_h/32, 10 + 1) * 32
            shape = (rand_h, rand_w)
        # Randomized n_neurons
        n_neurons = np.random.randint(2, 6 + 1)
        
        folder = os.path.join(synth_dir, "synth_{}neur_{:03d}".format(n_neurons, i))
        print("Creating stack %d/%d" % (i + 1, n_stacks), end="")
        print("  - folder:", folder)
        
        synth_stack, synth_seg = synthetic_stack(shape, n_images, n_neurons)
        
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "rgb_frames"), exist_ok=True)
        os.makedirs(os.path.join(folder, "seg_frames"), exist_ok=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Save full stacks
            io.imsave(os.path.join(folder, "RGB.tif"), to_npint(synth_stack))
            io.imsave(os.path.join(folder, "seg_ROI.tif"), to_npint(synth_seg))
            # Save image per image
            for j in range(n_images):
                io.imsave(os.path.join(folder, "rgb_frames", "rgb_{:04}.png".format(j)), to_npint(synth_stack[j]))
                io.imsave(os.path.join(folder, "seg_frames", "seg_{:04}.png".format(j)), to_npint(synth_seg[j]))
    
    duration = time.time() - start
    print("\nScript took {:02.0f}min {:02.0f}s.".format(duration // 60, duration % 60))
    
    # Launch the mask generation over the newly created synthetic dataset
    print("Launching weight generation script...")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.system("python %s --data_dir %s" % \
              (os.path.join(dir_path, "..", "deep_learning", "generate_weights.py"),
               synth_dir))