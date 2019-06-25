#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains a function to apply the CV detector to an RGB stack.
Created on Mon Dec 10 14:52:18 2018

@author: nicolas
"""

import numpy as np
from skimage import measure, filters
from skimage import morphology as morph
from skimage.morphology import disk

from axoid.utils.image import to_npint
from axoid.utils.ccreg import register_stack, shift_image
from axoid.utils.processing import nlm_denoising


def cv_detect(rgb_stack, h_red=11, h_green=11, sigma_gauss=2,
              thresholding_fn=filters.threshold_otsu, 
              registration=False, selem=disk(1), min_area = 6):
    """
    Use computer vision to detect ROI in given RGB stack.
    
    It first denoised the frames. Results is smoothed and then segmented by
    thresholding. Finally, an erosion is applied, followed by discarding small
    elements.
    
    Parameters
    ----------
    rgb_stack : ndarray
        Stack of RGB images where the first dimension is the time.
    h_red : int (default = 11)
        Parameter regulating the strength of the denoising in the red channel.
        See OpenCV documentation for more detail.
    h_green : int (default = 11)
        Parameter regulating the strength of the denoising in the green channel.
        See OpenCV documentation for more detail.
    sigma_gauss : float (default = 2)
        Sigma of the gaussian kernel for the smoothing.
    thresholding_fn : callable (default = threshold_otsu)
        Thresholding function for the segmentation. It should return a single value.
    registration : bool (default = False)
        If True, rgb_stack will be cross-correlation registered prior to denoising.
        Note that the return denoised stack will not be registered.
    selem : ndarray (default = disk(1))
        Array representing a pixel neighbourhood, for the morphological erosion.
    min_area : int (default = 6)
        ROI with an area under this value (in pixel) are discarded.
    
    Returns
    -------
    output : ndarray
        The binary detection array.
    """    
    stack = to_npint(rgb_stack)
    if registration:
        stack, reg_rows, reg_cols = register_stack(stack, channels=[0,1], return_shifts=True)
    
    denoised = nlm_denoising(stack, h_red=h_red, h_green=h_green)
    
    output = np.zeros(stack.shape[:-1], dtype=np.bool)
    for i in range(len(rgb_stack)):
        # Segmentation
        denoised_pp = filters.gaussian(denoised[i], sigma=sigma_gauss)
        seg = denoised_pp > thresholding_fn(denoised_pp)
        seg = morph.erosion(seg, selem=selem)
        
        # Mask creation
        mask = seg.copy()
        labels = measure.label(mask)
        for region in measure.regionprops(labels):
            if region.area < min_area: # discard small elements
                rows, cols = region.coords.T
                mask[rows, cols] = 0
        # Shift back the mask if the stack was registered
        if registration:
            mask = shift_image(mask, -reg_rows[i], -reg_cols[i])
        
        output[i] = mask
    return output
