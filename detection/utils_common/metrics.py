#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful metric functions for ROI detection.
Created on Mon Nov 12 09:37:00 2018

@author: nicolas
"""

import numpy as np
from skimage import measure


def loss_mae(predictions, targets, reduction='mean'):
    """Compute the (Mean) Average Error between predictions and targets.""" 
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return np.abs(targets - predictions).mean()
    elif reduction in ["sum"]:
        return np.abs(targets - predictions).sum()
    elif reduction in ["array", "no_reduction", "full"]:
        return np.abs(targets - predictions)
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def loss_l2(predictions, targets, reduction='mean'):
    """Compute the L2-norm loss between predictions and targets."""
    loss = []
    for i in range(len(targets)):
        loss.append(np.linalg.norm(targets[i] - predictions[i]))
    
    if reduction in ["elementwise_mean", "mean", "ave", "average"]:
        return np.mean(loss)
    elif reduction in ["sum"]:
        return np.sum(loss)
    elif reduction in ["array", "no_reduction", "full"]:
        return np.array(loss)
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def dice_coef(predictions, targets, reduction='mean'):
    """Compute the Dice coefficient between predictions and targets."""
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
    elif reduction in ["array", "no_reduction", "full"]:
        return np.array(dice)
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)

def crop_metric(metric_fn, predictions, targets, scale=4.0, reduction='mean'):
    """
    Compute the metric around the cropped targets' connected regions.
    
    Size of the cropped region will be bounding_box * scale.
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
    elif reduction in ["array", "no_reduction", "full"]:
        return np.array(metric)
    else:
        raise ValueError("""Unknown reduction method "%s".""" % reduction)