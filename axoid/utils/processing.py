#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for processing stacks of images.
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
    """
    Horizontal line element for morpholgical operations.
    
    Parameters
    ----------
    length : int
        Length in pixel of the line.
    
    Returns
    -------
    selem : ndarray
        Square array of shape (length, length), with zeroes everywhere but at
        the middle row.
    """
    selem = np.zeros((length, length), dtype=np.uint8)
    selem[int(length / 2), :] = 1
    return selem

def vline(length):
    """
    Vertical line element for morpholgical operations.
    
    Parameters
    ----------
    length : int
        Length in pixel of the line.
    
    Returns
    -------
    selem : ndarray
        Square array of shape (length, length), with zeroes everywhere but at
        the middle column.
    """
    selem = np.zeros((length, length), dtype=np.uint8)
    selem[:, int(length / 2)] = 1
    return selem

def identity(stack, selem=None):
    """Identity function, return a copy of the stack."""
    return stack.copy()

def median_filter(stack, selem=None):
    """
    Apply median filtering to all images in the stack.
    
    Parameters
    ----------
    stack : ndarray
        Stack of images. If dtype is not uint8 or uint16, it will be casted
        before computing the median, and then casted back.
    selem : ndarray
        Array representing the neighbourhood of pixels on which to compute the 
        median.
    
    Returns
    -------
    filtered_stack : ndarray
        Resulting stack after median filtering of the individual images.
    """
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
    """
    Apply morphological opening to all images in the stack.
    
    Parameters
    ----------
    stack : ndarray
        Stack of images, should be greyscale.
    selem : ndarray
        Array representing the neighbourhood of pixels on which to compute the 
        opening.
    
    Returns
    -------
    filtered_stack : ndarray
        Resulting stack after morphological opening of the individual images.
    """   
    filtered_stack = np._like(stack)
    
    for i in range(len(stack)):
        filtered_stack[i] = morph.opening(stack[i], selem=selem)
        
    return filtered_stack

def preprocess_stack(stack):
    """
    Apply the median filtering, then morphological opening to the stack.
    
    /!\ Deprecated
    Pre-processing function used in older version of the project. It is kept
    as I am not sure if it is still used somewhere else.
    
    Parameters
    ----------
    stack : ndarray
        Stack of images, should be greyscale.
    
    Returns
    -------
    filtered_stack : ndarray
        Resulting stack after pre-processing of the individual images.
    """
    filtered_stack = median_filter(stack, disk(1))
    return morph_open(filtered_stack, disk(1))

def flood_fill(image, fill_val=1):
    """
    Fill the contours in image using openCV's flood-fill algorithm.
    
    Parameters
    ----------
    image : ndarray
        Image with holes to fill. It will be casted to uint8 for this.
    fill_val : int (default = 1)
        Value to place in the holes after filling. Should be between 0 and 255.
    
    Returns
    -------
    filled_image : ndarray
        Image as uint8 with filled holes.
    """
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
    """
    Fuse labelled objects smaller than min_area to other connected ones.
    
    It first fuses all small objects that are touching together, then fuse the 
    remaining small objects to touching objects regardless of these objects'
    size (it takes the smallest among them).
    If small isolated objects remaind, they are not removed.
    
    Parameters
    ----------
    label : ndarray
        Label image where 0 is background, and each label corresponds to one 
        object.
    min_area : int
        Minimum area in pixel under which an object is considered small.
    connectivity : int (default = 1)
        Type of connectivity to consider objects touching. 1 is 4-neighbour,
        and 2 is 8-neighbour.
    
    Returns
    -------
    label_out : ndarray
        Label image where small objects are fused.
        Note that the label are renumbered to start at 1, so they are not 
        conserved from input.
    """
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
    """
    Apply Non-Local means denoising to the stack, or the image.
    
    /!\ It takes some time (a few minutes) for full stacks.
    
    Parameters
    ----------
    rgb_stack : ndarray
        Stack of RGB images where the first dimension is the time.
    img_id : int (optional)
        Index of a frame to denoise. If given, only this frame will be denoised.
        Else, all rgb_stack will be denoised.
    h_red : int (default = 11)
        Parameter regulating the strength of the denoising in the red channel.
        See OpenCV documentation for more detail.
    h_green : int (default = 11)
        Parameter regulating the strength of the denoising in the green channel.
        See OpenCV documentation for more detail.
    registration : bool (default = False)
        If True, rgb_stack will be cross-correlation registered prior to denoising.
        Note that the return denoised stack will not be registered.
    reg_ref : int (default = 0)
        If registration is used, this is the index of the frame to use as 
        reference for the registration.
    return_rgb : bool (default = False)
        If True, the return stack will be RGB with the channel individually
        denoised (note that green and blue will be identical).
        If False, a greyscale average is returned.
    
    Returns
    -------
    denoised : ndarray
        The denoised array. Can be RGB or greyscale depending on return_rgb.
    """
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
