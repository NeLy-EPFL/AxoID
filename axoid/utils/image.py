#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for loading/saving/displaying stacks and images.
Created on Mon Oct  1 17:48:29 2018

@author: nicolas
"""

import numpy as np
from matplotlib import cm
from skimage import io, color, measure


def imread_to_float(filename, scaling=None, return_scaling=False):
    """
    Return the loaded stack/image from filename, casted to float32 and rescaled.
    
    Parameters
    ----------
    filename : str
        Path to the image file with its extension.
    scaling : int or float (optional)
        The image will be rescaled as image / scaling.
        If not given, scaling will be computed as max(image).
    return_scaling : bool (default = False)
        If True, the scaling is returned.
    
    Returns
    -------
    stack : ndarray
        Image or stack of images loaded as an numpy float32 array.
    [scaling] : int or float (optional)
        Value used for scaling the images.
    """
    stack = np.asarray(io.imread(filename))
    stack = np.nan_to_num(stack)
    # If no scaling is precised, use max value of stack
    if scaling is None:
        scaling = stack.max()
    stack = stack.astype(np.float32) / scaling
    if return_scaling:
        return stack, scaling
    else:
        return stack

def to_npint(stack, dtype=np.uint8, float_scaling=None):
    """
    Scale and cast the stack/image to given integer type.
    
    This function is useful for images which have value range of [0, 1] in float,
    but should be converted to integer with different range (e.g. [0, 255] for
    np.uint8).
    
    Parameters
    ----------
    stack : ndarray
        Image or stack of images to convert.
    dtype : numpy.dtype (default = np.uint8)
        Data type in which the stack will be converted. Should be integer.
    float_scaling : float (optional)
        If stack has float dtype, it will be scaled up by this value.
        If not given, float_scaling is set to the maximum allowed by dtype.
    
    Returns
    -------
    stack_int : ndarray
        Stack casted to dtype, with a rescaling.
    """
    if stack.dtype == dtype:
        return stack.copy()
        
    if issubclass(stack.dtype.type, np.floating):
        # If no scaling is precised, use max range of given type
        if float_scaling is None:
            float_scaling = np.iinfo(dtype).max
        stack_int = (stack * float_scaling).astype(dtype)
    elif stack.dtype == np.bool:
        # If boolean, we set 1 to max range of dtype (e.g., 255 for np.uint8)
        stack_int = stack.astype(dtype) * np.iinfo(dtype).max
    else:
        stack_int = stack.astype(dtype)
    return stack_int

def to_id_cmap(image, cmap="viridis", vmin=0.99, vmax=None):
    """
    Apply the color map to the identity image.
    
    Parameters
    ----------
    image : ndarray
        Identity image, of integer dtype. 0 should be background, 1 and above
        corresponds to different identities.
    cmap : str (default = "viridis")
        Name of the matplotlib colormap to use.
    vmin : int or float (default = 0.99)
        Minimum value to use for the color map. Recommended value is just below
        the lowest identity, but above the background value.
    vmax : int or float (optional)
        Maximum value to use for the color map. If not given, will take image.max().
    
    Returns
    -------
    RGB image corresponding to image convert to the colormap.
    """
    id_cmap = cm.get_cmap(cmap)
    id_cmap.set_under([0,0,0])
    if vmax is None:
        vmax = image.max()
    
    out = image.astype(np.float)
    out = id_cmap((out - vmin) / (vmax - vmin))[...,:-1]
    return to_npint(out)

def gray2red(image):
    """
    Create an RGB image with image in the red channel, and 0 in the others.
    
    /!\ It does not verify if image is grayscale in order to work with stacks 
    of images too!
    
    Parameters
    ----------
    image : ndarray
        Image or stack of images in greyscale to convert to RGB, with 0 in 
        green and blue.
    
    Returns
    -------
    RGB image with red values equal to image, and 0 in green and blue.
    """
    red = image
    green = np.zeros_like(image)
    blue = np.zeros_like(image)
    return np.stack([red, green, blue], axis=-1)

def rg2gray(image):
    """
    Create a grayscale images with the red and green channel of the image.
    
    Parameters
    ----------
    image : ndarray
        Image or stack of images in RGB to convert to greyscale without blue channel.
    
    Returns
    -------
    Greyscale image with original blue channel ignored.
    """
    # Use weights from contemporary CRT phosphors (see skimage.color.rgb2gray doc)
    weights = np.array([0.2125, 0.7154, 0.0], image.dtype)
    return image @ weights / weights.sum()

def overlay_mask(image, mask, opacity=0.25, mask_color=[1.0, 0.0, 0.0], rescale_img=False):
    """
    Merge the mask as an overlay over the image.
    
    Parameters
    ----------
    image : ndarray
        Image on which the mask is to be overlayed.
    mask : ndarray
        Boolean mask to be overlayed over image.
    opacity : float (default = 0.25)
        Opacity (=alpha) of the overlay. 0 is invisible, 1 is opaque.
    mask_color : array-like (default = [1, 0, 0])
        RGB value of the overlay.
    rescale_image : bool (default = False)
        If True, the image will be rescaled as image / image.max()
    
    Returns
    -------
    overlay : ndarray
        Overlay of mask over image.
    """
    mask_color = np.array(mask_color, dtype=np.float32)
    if image.ndim == 2:
        overlay = color.gray2rgb(image)
    else:
        overlay = image.copy()
    
    overlay = overlay.astype(np.float)
    if rescale_img:
        overlay /= overlay.max()
        
    overlay[mask.astype(np.bool), :] *= 1 - opacity
    overlay[mask.astype(np.bool), :] += mask_color * opacity
    return overlay.astype(image.dtype)

def overlay_mask_stack(stack, mask, opacity=0.25, mask_color=[1.0, 0.0, 0.0], rescale_img=False):
    """
    Merge the mask as an overlay over the stack.
    
    Same as overlay_mask(), but for stack of images.
    """
    mask_color = np.array(mask_color, dtype=np.float32)
    if stack.ndim == 3:
        overlay = color.gray2rgb(stack)
    else:
        overlay = stack.copy()
        
    overlay = overlay.astype(np.float)
    for i in range(len(stack)):
        overlay[i] = overlay_mask(overlay[i], mask[i], opacity=opacity, 
               mask_color=mask_color, rescale_img=rescale_img)
    return overlay.astype(stack.dtype)

def overlay_preds_targets(predictions, targets, masks=None):
    """
    Create an image with prediction and target (and mask) for easy comparison.
    
    True positives appear green,
    false positives appear white,
    false negatives appear red,
    true negatives appear black.
    Masks appear as transparent yellow.
    
    Parameters
    ----------
    predictions : ndarray
        Image or stack of image (binary) representing the predictions.
    targets : ndarray
        Same but for the target binary images.
    masks : ndarray (optional)
        Same but for masks (see loss masking).
    
    Returns
    -------
    final : ndarray
        RGB array of the overlay.
    """
    # Select correct overlay function in order to work with image, and stack
    if predictions.ndim == 3:
        overlay_fn = overlay_mask_stack
    else:
        overlay_fn = overlay_mask
    
    # Add predicted annotations as green
    correct = overlay_fn(predictions, np.logical_and(predictions, targets), 
                         opacity=1, mask_color=[0,1,0])
    # Add missed annotations as red
    incorrect = overlay_fn(correct, np.logical_and(targets, np.logical_not(predictions)), 
                           opacity=1, mask_color=[1,0,0])
    if masks is None:
        final = incorrect
    else: # add masks as transparent yellow
        final = overlay_fn(incorrect, masks, opacity=0.5, mask_color=[1,1,0])
    return final

def overlay_contours(image, mask, rescale_img=False):
    """
    Put the contours of mask over image.
    
    Contours are displayed in white,
    
    Parameters
    ----------
    image : ndarray
        Image on which the mask contours are to be overlayed.
    mask : ndarray
        Boolean mask from which contours are to be overlayed over image.
    rescale_image : bool (default = False)
        If True, the image will be rescaled as image / image.max()
    
    Returns
    -------
    overlay : ndarray
        Overlay of mask contours over image.
    """
    contour = np.zeros(mask.shape, dtype=np.bool)
    coords = measure.find_contours(mask, 0.5) # 0.5 to work with both float and int
    for coord in coords:
        rows = np.rint(coord[:,0]).astype(np.int)
        cols = np.rint(coord[:,1]).astype(np.int)
        contour[rows, cols] = True
    return overlay_mask(image, contour, opacity=1.0, mask_color=[1,1,1], rescale_img=rescale_img)

def overlay_contours_stack(stack, mask, rescale_img=False):
    """
    Put the contours of mask over image.
    
    Same as overlay_contours(), but for stack of images.
    """
    contour = np.zeros(mask.shape, dtype=np.bool)
    for i in range(len(mask)):
        coords = measure.find_contours(mask[i], 0.5) # 0.5 to work with both float and int
        for coord in coords:
            rows = np.rint(coord[:,0]).astype(np.int)
            cols = np.rint(coord[:,1]).astype(np.int)
            contour[i, rows, cols] = True
    return overlay_mask_stack(stack, contour, opacity=1.0, mask_color=[1,1,1], rescale_img=rescale_img)