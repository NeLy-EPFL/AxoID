#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for computing metrics with PyTorch.
Created on Tue Nov 27 13:57:06 2018

@author: nicolas
"""

import torch

from axoid.utils.metrics import dice_coef, crop_metric

   
def get_dice_metric(reduction='mean'):
    """
    Return a metric function that computes the dice coefficient.
    
    Parameters
    ----------
    reduction : str (default = 'mean')
        Reduction scheme for the loss, see axoid.utils.metrics for details.
        Briefly, 'mean' return the average loss of the batch element.
    
    Returns
    -------
    dice_metric : callable
        Function taking predictions and targets, and returning their dice
        coefficient.
    """
    return lambda preds, targets: torch.tensor(
            dice_coef((torch.sigmoid(preds.cpu()) > 0.5).detach().numpy(),
                      targets.cpu().detach().numpy(),
                      reduction=reduction))

def get_crop_dice_metric(scale=4.0, reduction='mean'):
    """
    Return a metric function that computes the cropped dice coefficient.
    
    Parameters
    ----------
    scale : float (default = 4)
        Scaling of the cropped regions. The regions are square crops around the
        targets' ROIs, where their size is scale * bounding-box.
    reduction : str (default = 'mean')
        Reduction scheme for the loss, see axoid.utils.metrics for details.
        Briefly, 'mean' return the average loss of the batch element.
    
    Returns
    -------
    dice_metric : callable
        Function taking predictions and targets, and returning their cropped
        dice coefficient.
    """
    return lambda preds, targets: torch.tensor(
            crop_metric(dice_coef,
                        (torch.sigmoid(preds.cpu()) > 0.5).detach().numpy(),
                        targets.cpu().detach().numpy(),
                        scale=scale,
                        reduction=reduction))
