#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for computing metrics with PyTorch.
Created on Tue Nov 27 13:57:06 2018

@author: nicolas
"""

import torch

from utils_common.metrics import dice_coef, crop_metric

   
def get_dice_metric(reduction='mean'):
    """Return a metric function that computes the dice coefficient."""
    return lambda preds, targets: torch.tensor(
            dice_coef((torch.sigmoid(preds.cpu()) > 0.5).detach().numpy(),
                      targets.cpu().detach().numpy(),
                      reduction=reduction))

def get_crop_dice_metric(scale=4.0, reduction='mean'):
    """Return a metric function that computes the cropped dice coefficient."""
    return lambda preds, targets: torch.tensor(
            crop_metric(dice_coef,
                        (torch.sigmoid(preds.cpu()) > 0.5).detach().numpy(),
                        targets.cpu().detach().numpy(),
                        scale=scale,
                        reduction=reduction))