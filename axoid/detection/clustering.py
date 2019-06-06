#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module with functions used for selecting subsets of similar frames to use for 
finetuning.
Created on Thu Jun  6 11:04:39 2019

@author: nicolas
"""

import warnings

import numpy as np
from skimage import filters, measure, morphology, segmentation
from sklearn.cluster import OPTICS

from axoid.utils.image import rg2gray
from axoid.utils.processing import fuse_small_objects


def segment_projection(projection, min_area=None, separation_border=True):
    """Segment the ROIs of the projection image."""
    # Binary segmentation by local thersholding
    bin_projection = rg2gray(projection)
    bin_projection = bin_projection > filters.threshold_local(bin_projection, 25, offset=-0.05)
    if min_area is not None:
        # Delete small ROIs
        small_rois = np.logical_xor(bin_projection, morphology.remove_small_objects(bin_projection, min_area))
        bin_projection = np.logical_xor(bin_projection, small_rois)

    if separation_border:    
        # Peaks detection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            peaks = morphology.h_maxima(rg2gray(projection), 0.05)
            peaks = segmentation.clear_border(peaks)
            peaks *= bin_projection # remove peaks outside of segmentation

        # Watershed to separate touching axons
        markers = measure.label(peaks, connectivity=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            watershed = morphology.watershed(-rg2gray(projection), markers, mask=bin_projection)

        if min_area is not None:
            # Fuse touching small areas
            post_fusion = fuse_small_objects(watershed, min_area)

        # Add border between touching labels
        sep_labels = post_fusion.copy()
        for i in np.unique(post_fusion):
            if i == 0:
                continue
            # use post_fusion for a 2 pixel-wide separation, or sep_labels for a 1 pixel-wide one
            sep_labels[np.logical_and(morphology.dilation(sep_labels == i), sep_labels != i)] = 0

        # Convert to boolean
        seg_projection = sep_labels.astype(np.bool)
    else:
        seg_projection = bin_projection
    
    return seg_projection


def crosscorr_matrix(stack, normalize=True, discard_blue=True, keep_diag=False):
    """Return the cross-correlation matrix of the stack images."""
    # Normalize the images
    if normalize:
        images = (stack - stack.mean((1,2), keepdims=True)) / stack.std((1,2), keepdims=True)
    else:
        images = stack.copy()
    
    # Set blue to 0 to avoid counting multiple times GCaMP channel
    # Following is faster with this rather than simply discarding the last channel,
    # I don't know why...
    if discard_blue:
        images[..., 2] = 0
       
    # Flatten each image into a vector, and then simply do a matrix 
    flat_images = images.reshape((len(images), -1))
    cc_matrix = flat_images @ flat_images.transpose()
    
    # Set diagonal to nan
    if not keep_diag:
        cc_matrix[np.eye(len(images), dtype=np.bool)] = np.nan
    
    return cc_matrix

def cluster_matrix(matrix):
    """Return the labels of the clustered similarity matrix."""
    # Set nan to 0 to avoid problem in clustering
    dist = np.nan_to_num(matrix)
    # Transform similarity matrix in distance matrix by inverting similarity
    # as (min, max) --> (max, min)
    dist = dist.max() - dist
    
    optics = OPTICS(min_samples=20, max_eps=np.median(dist)/2, metric="precomputed")
    labels = optics.fit_predict(dist)
    
    return labels

def cluster_crosscorr(matrix, labels, cluster_label):
    """Return the average cross-correlation of the cluster frames."""
    idx_cluster = np.where(labels == cluster_label)[0]
    submatrix = matrix[idx_cluster][:, idx_cluster]
    mean_cc = np.nanmean(submatrix)
    return mean_cc

def similar_frames(stack):
    """Return the indices of the most similar frames (in the cross-correlation sense)."""
    # Compute the cross-correlation matrix
    cc_matrix = crosscorr_matrix(stack)
    
    # Cluster it
    labels = cluster_matrix(cc_matrix)
    
    # Find cluster with highest average cross-correlation
    cluster_label = -1
    cluster_cc = -np.inf
    for l in np.unique(labels):
        # Ignore outliers
        if l == -1:
            continue
        new_cc = cluster_crosscorr(cc_matrix, labels, l)
        if new_cc > cluster_cc:
            cluster_cc = new_cc
            cluster_label = l
    
    # Return indices of the cluster frames
    return np.where(labels == cluster_label)[0]