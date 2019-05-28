#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for processing stacks.
Created on Mon Nov 12 09:28:07 2018

@author: nicolas
"""

import warnings
import numpy as np
from skimage import filters, measure
from skimage import morphology as morph
from skimage.morphology import disk
import cv2

from .image import to_npint
from .ccreg import register_stack
from .multithreading import run_parallel

def hline(length):
    """Horizontal line element for morpholgical operations."""
    selem = np.zeros((length, length), dtype=np.uint8)
    selem[int(length / 2), :] = 1
    return selem

def vline(length):
    """Vertical line element for morpholgical operations."""
    selem = np.zeros((length, length), dtype=np.uint8)
    selem[:, int(length / 2)] = 1
    return selem

def identity(stack, selem=None):
    """Identity function, return the same stack."""
    return stack

def median_filter(stack, selem=None):
    """Apply median filtering to all images in the stack."""
    # Median works with uint8 or 16
    if stack.dtype in [np.uint8, np.uint16]:
        median_type = stack.dtype
    else:
        median_type = np.uint8
    
    # Median filtering with disabled warnings
    filtered_stack = np.zeros(stack.shape, dtype=median_type)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(stack)):
            filtered_stack[i] = filters.median(stack[i], selem=selem)
    
    # Cast back if needed
    if median_type != stack.dtype:
        # if uint32 or 64, can simply cast
        if isinstance(stack.dtype, np.unsignedinteger):
            filtered_stack = filtered_stack.astype(stack.dtype)
        # if float, change range to [0,1]
        elif stack.dtype in [np.float16, np.float32, np.float64, np.float128]:
            filtered_stack = (filtered_stack / 255).astype(stack.dtype)
        else:
            print("Warning: Unable to cast back to %s after median filtering. "
                  "Returning an np.uint8 array." % stack.dtype)
    
    return filtered_stack

def morph_open(stack, selem=None):
    """Apply morphological opening to all images in the stack."""   
    filtered_stack = np.zeros(stack.shape, dtype=stack.dtype)
    
    for i in range(len(stack)):
        filtered_stack[i] = morph.opening(stack[i], selem=selem)
        
    return filtered_stack

def preprocess_stack(stack):
    """Apply the preprocessing function to the stack."""
    filtered_stack = median_filter(stack, disk(1))
    return morph_open(filtered_stack, disk(1))

def flood_fill(image, fill_val=1):
    """Fill the contours in image using openCV's flood-fill algorithm."""
    image_out = image.astype(np.uint8)
    
    # Mask used to flood fill
    height, width = image.shape
    mask = np.zeros((height + 2, width + 2), np.uint8)
    
    # Flood fill (in-place) from point (0,0)
    cv2.floodFill(image_out, mask, (0,0), 1)
    
    # Invert filled image
    mask = image_out == 0
    
    # Combine contours with filled ROI
    image_out = image.astype(np.uint8)
    image_out[mask] = fill_val
    
    return image_out

def _connected(label, region1, region2, connectivity):
    """Return the connectivity state between the 2 regions of the label image."""
    num_obj1 = measure.label(label == region1.label, connectivity=connectivity,
                             return_num=True)[1]
    num_obj2 = measure.label(label == region2.label, connectivity=connectivity,
                             return_num=True)[1]
    num_obj_tot = measure.label(label == region1.label, connectivity=connectivity,
                                return_num=True)[1]
    # If there are less objects in the combined frames, the regions must be connected
    if num_obj_tot < num_obj1 + num_obj2:
        return True
    else:
        return False
def fuse_small_objects(label, min_area, connectivity=1):
    """Fuse labelled objects smaller than min_area to other connected ones."""
    regions = measure.regionprops(label)
    small_regions = [region for region in regions if region.area < min_area]
    if len(small_regions) == 0:
        label_out =  measure.label(label, connectivity=connectivity)
        return label_out
    
    # Create an image with fused connected small regions
    fused_small = np.sum([label == reg.label for reg in small_regions], axis=0)
    fused_small = measure.label(fused_small, connectivity=connectivity)
    
    # Create an image with large regions + fused small
    label_out = label.copy()
    label_out[fused_small != 0] = fused_small[fused_small != 0] + label_out.max()
    
    # Fuse remaining small areas with their smallest neighbouring region
    regions = measure.regionprops(label_out)
    small_regions = [region for region in regions if region.area < min_area]
    for region in small_regions:
        # Look for neighbours <=> connected regions
        neighbours = [] # store (label, area) of each neighbour as a dict
        for region2 in regions:
            if region.label == region2.label:
                continue
            if _connected(label_out, region, region2, connectivity):
                neighbours.append({"label": region2.label, "area": region2.area})
        # Apply label of smallest neighbour
        if len(neighbours) > 0:
            sorted_neighb = sorted(neighbours, key=lambda n: n["area"])
            label_out[label_out == region.label] = sorted_neighb[0]["label"]
    
    # Make unique labels starting at 1
    label_out = measure.label(label_out, connectivity=connectivity)
    return label_out

def nlm_denoising(rgb_stack, img_id=None, h_red=11, h_green=11, 
                  registration=False, reg_ref=0,
                  return_rgb=False):
    """Apply Non-Local means denoising to the stack, or the specific image if 
    img_id is given."""
    temporal_window_size = 5
    search_window_size = 21
    
    stack = to_npint(rgb_stack)
    if registration:
        stack = register_stack(stack, ref_num=reg_ref)
    
    # Loop the stack so that masks can be made for first and last images
    loop_stack = np.concatenate((stack[- (temporal_window_size - 1)//2:],
                                 stack, 
                                 stack[:(temporal_window_size - 1)//2]))
    
    # Denoise each channel
    def denoise_stack(channel_num, h_denoise):
        """Denoise selected channel from loop_stack (function used for parallelization)."""
        loop_channel = loop_stack[..., channel_num]
        if img_id is not None:
            denoised = cv2.fastNlMeansDenoisingMulti(loop_channel, img_id + (temporal_window_size - 1)//2, 
                     temporal_window_size, None, h_denoise, 7, search_window_size)
        else:
            denoised = np.zeros(stack[...,0].shape, dtype=loop_channel.dtype)
            for i in range(len(stack)):
                denoised[i] = cv2.fastNlMeansDenoisingMulti(loop_channel, i + (temporal_window_size - 1)//2, 
                        temporal_window_size, None, h_denoise, 7, search_window_size)
        return denoised
    
    denoised_r, denoised_g = run_parallel(
        lambda: denoise_stack(0, h_red),
        lambda: denoise_stack(1, h_green)
    )
    
    if return_rgb:
        denoised = np.stack([denoised_r, denoised_g, denoised_g], axis=-1)
    else:
        denoised = np.maximum(denoised_r, denoised_g)
    
    if issubclass(rgb_stack.dtype.type, np.floating):
        denoised = (denoised / 255).astype(rgb_stack.dtype)
    return denoised
