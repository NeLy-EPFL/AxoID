#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful metric functions.
Created on Mon Nov 12 09:37:00 2018

@author: nicolas
"""

import numpy as np
from skimage import measure


#%% Metrics for ROI binary detection

def loss_mae(predictions, targets, reduction='mean'):
    """
    Compute the (Mean) Average Error between predictions and targets.
    
    Parameters
    ----------
    predictions : ndarray
        Array of the predictions. Can be multi-dimensional.
    targets : ndarray
        Array of the targets. Same dimensions as predictions.
    reduction : str (default = "mean")
        Type of reduction to apply to the loss:
           - "elementwise_mean", "mean", "ave", "average": average of the single
           element's loss
           - "sum": sum of the single element's loss
           - None, "none", "array", "no_reduction", "full": array of each
           element's loss
    
    Returns
    -------
    loss : float or ndarray
        MAE loss of the predictions compared to targets. See reduction for 
        details of the returned value.
    """ 
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return np.abs(targets - predictions).mean()
    elif reduction in ["sum"]:
        return np.abs(targets - predictions).sum()
    elif reduction in [None, "none", "array", "no_reduction", "full"]:
        return np.abs(targets - predictions)
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def loss_l2(predictions, targets, reduction='mean'):
    """
    Compute the L2-norm loss between predictions and targets.
    
    Parameters
    ----------
    predictions : ndarray
        Array of the predictions. Can be multi-dimensional.
    targets : ndarray
        Array of the targets. Same dimensions as predictions.
    reduction : str (default = "mean")
        Type of reduction to apply to the loss:
           - "elementwise_mean", "mean", "ave", "average": average of the single
           element's loss
           - "sum": sum of the single element's loss
           - None, "none", "array", "no_reduction", "full": array of each
           element's loss
    
    Returns
    -------
    loss : float or ndarray
        L2-norm loss of the predictions compared to targets. See reduction for 
        details of the returned value.
    """
    loss = []
    for i in range(len(targets)):
        loss.append(np.linalg.norm(targets[i] - predictions[i]))
    
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return np.mean(loss)
    elif reduction in ["sum"]:
        return np.sum(loss)
    elif reduction in [None, "none", "array", "no_reduction", "full"]:
        return np.array(loss)
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def dice_coef(predictions, targets, reduction='mean'):
    """
    Compute the Dice coefficient between predictions and targets.
    
    Parameters
    ----------
    predictions : ndarray
        Array of the predictions. Should be NxHxW.
    targets : ndarray
        Array of the targets. Same dimensions as predictions.
    reduction : str (default = "mean")
        Type of reduction to apply to the loss:
           - "elementwise_mean", "mean", "ave", "average": average of the single
           element's loss
           - "sum": sum of the single element's loss
           - None, "none", "array", "no_reduction", "full": array of each
           element's loss
    
    Returns
    -------
    coef : float or ndarray
        Dice coefficient of the predictions compared to targets. See reduction for 
        details of the returned value.
    """
    dice = []
    for i in range(len(targets)):
        total_pos = targets[i].sum() + predictions[i].sum()
        if total_pos == 0: # No true positive, and no false positive --> correct
            dice.append(1.0)
        else:
            dice.append(2.0 * np.logical_and(targets[i], predictions[i]).sum() / total_pos)
    
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return np.mean(dice)
    elif reduction in ["sum"]:
        return np.sum(dice)
    elif reduction in [None, "none", "array", "no_reduction", "full"]:
        return np.array(dice)
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def crop_metric(metric_fn, predictions, targets, scale=4.0, reduction='mean'):
    """
    Compute the metric around the cropped targets' connected regions.
    
    This works for binary images, where the loss should only be taken around
    target's regions.
    Size of the cropped region will be bounding_box * scale.
    
    Parameters
    ----------
    metric_fn : callable
        Loss function which takes (predictions, targets, reduction) as input,
        and return a loss.
    predictions : ndarray
        Array of the predictions. Should be NxHxW.
    targets : ndarray
        Array of the targets. Same dimensions as predictions.
    reduction : str (default = "mean")
        Type of reduction to apply to the loss:
           - "elementwise_mean", "mean", "ave", "average": average of the single
           element's loss
           - "sum": sum of the single element's loss
           - None, "none", "array", "no_reduction", "full": array of each
           element's loss
    
    Returns
    -------
    loss : float or ndarray
        Loss of the predictions compared to targets. See reduction for 
        details of the returned value.
    """
    metric = []
    n_no_positive = 0 # number of target with no positive pixels (fully background)
    for i in range(len(targets)):
        labels = measure.label(targets[i])
        # If no true positive region, does not consider the image
        if labels.max() == 0:
            n_no_positive += 1
            continue
        regionprops = measure.regionprops(labels)
        
        mask = np.zeros(targets[i].shape, dtype=np.bool)
        # Loop over targets' connected regions
        for region in regionprops:
            min_row, min_col, max_row, max_col = region.bbox
            height = max_row - min_row
            width = max_col - min_col
            max_row = int(min(targets[i].shape[0], max_row + height * (scale-1) / 2))
            min_row = int(max(0, min_row - height * (scale-1) / 2))
            max_col = int(min(targets[i].shape[1], max_col + width * (scale-1) / 2))
            min_col = int(max(0, min_col - width * (scale-1) / 2))
            mask[min_row:max_row, min_col:max_col] = True
            
        metric.append(metric_fn(np.array([predictions[i][mask]]), 
                                np.array([targets[i][mask]])))
    
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return np.sum(metric) / (len(targets) - n_no_positive)
    elif reduction in ["sum"]:
        return np.sum(metric)
    elif reduction in [None, "none", "array", "no_reduction", "full"]:
        return np.array(metric)
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)


#%% Misc. metrics

def gradient_norm(image):
    """
    Return the norm of the gradient of the image.
    
    Defined as the Frobenius norm of the gradient image, where the gradient is 
    constructed as the norm of the gradient at each pixel.
    
    Parameters
    ----------
    image : ndarray
        Greyscale image on which to compute the gradient norm.
    
    Returns
    -------
    grad_norm : float
        Norm of the gradient of image.
    """
    if image.ndim != 2:
        raise ValueError("image should be 2-dimensional (image.ndim=%d)" % image.ndim)
        
    # Compute the gradient image
    grad = np.gradient(image)
    grad = np.linalg.norm(grad, axis=0)
    
    # Return its Frobenius norm
    return np.linalg.norm(grad, ord='fro')