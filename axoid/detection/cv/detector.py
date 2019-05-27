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

from ...utils.image import to_npint
from ...utils.register_cc import register_stack, shift_image
from ...utils.processing import nlm_denoising


def cv_detect(rgb_stack, h_red=11, h_green=11, sigma_gauss=2,
              thresholding_fn=filters.threshold_otsu, 
              registration=False, selem=disk(1)):
    """Use computer vision to detect ROI in given RGB stack."""
    min_area = 6 # minimum area in pixel for an ROI
    
    
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
