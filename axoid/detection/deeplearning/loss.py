#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for computing losses with PyTorch.
Created on Wed Feb 20 10:29:37 2019

@author: nicolas
"""

import torch

   
def get_BCEWithLogits_loss(reduction='mean', 
                           pos_weight=1, neg_weight=1):
    """
    Return a loss function that computes the binary cross entropy with logits,
    with pixel-wise weighting.
    
    The weights in the ROIs are constant and equal to pos_weight.
    The ones in the background are set to neg_weight + (pos_weight - neg_weight) * weight,
    where weight is the weighting mask (if inexistent, is set to 0).
    """
    # Create the loss function
    def loss_fn(prediction, target, weight=None):
        # Create and rescale weights
        if weight is None:
            weight = torch.zeros_like(target)
        else:
            weight *= pos_weight - neg_weight
        weight += neg_weight
        weight[target.byte()] = pos_weight
        
        # Compute BCE
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction, target, weight=weight, reduction=reduction)
        return loss
    
    return loss_fn