#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for fine tuning with PyTorch.
Created on Mon Apr  8 14:13:08 2019

@author: nicolas
"""

import copy
import numpy as np

import torch

from .utils_data import pad_collate, normalize_range, pad_transform_stack, compute_weights
from .utils_loss import get_BCEWithLogits_loss
from .utils_metric import get_dice_metric
from .utils_test import evaluate_stack


def fine_tune(model, inputs, annotations, weights=None, n_iter=200, n_valid=1,
              batch_size=16, learning_rate = 0.0005, verbose=1):
    """Fine tune the given model on the annotated data, and return the resulting model."""
    u_depth = len(model.convs)
    device = model.device
    annotated_per_batch = min(len(annotations) - n_valid, batch_size) # number of annotated frames in each batch
    metrics = {"dice": get_dice_metric()}
    eval_transform = lambda stack: normalize_range(pad_transform_stack(stack, u_depth))
    
    # Compute class weights (on train data) and pixel-wise weighting images
    pos_count = (annotations[:len(annotations) - n_valid] == 1).sum()
    neg_count = (annotations[:len(annotations) - n_valid] == 0).sum()
    pos_weight = torch.tensor((neg_count + pos_count) / (2 * pos_count)).to(device)
    neg_weight = torch.tensor((neg_count + pos_count) / (2 * neg_count)).to(device)
    if weights is None:
        weights = compute_weights(annotations)
    
    # Make a copy of the model, and keep track of the best state_dict
    model_ft = copy.deepcopy(model)
    if n_valid > 0:
        best_state_dict = model_ft.state_dict()
    
    # Define loss and optimizer
    loss_fn = get_BCEWithLogits_loss(pos_weight=pos_weight, neg_weight=neg_weight)
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=learning_rate)
    
    # Set model to training mode
    model_ft.train()
    
    # Iterate over the data
    if verbose:
        print("Iteration (over %d): " % n_iter)
    if n_valid > 0:
        best_iter, best_dice = -1, -1
    for i in range(n_iter):
        # Randomly select elements
        rand_idx = np.random.choice(np.arange(len(annotations) - n_valid), 
                                    size=annotated_per_batch, replace=False)
        # Keep only relevant input channels
        images = np.stack([inputs[rand_idx,:,:,0], inputs[rand_idx,:,:,1]], axis=1)
        # Apply train transforms
        images = normalize_range(pad_transform_stack(images, u_depth))
        targets = pad_transform_stack(annotations[rand_idx], u_depth)
        weights_batch = pad_transform_stack(weights[rand_idx], u_depth)
        
        # Extract items from batch and send to model device
        items_annotated = [(i, t, w) for i, t, w in zip(images, targets, weights_batch)]
        batch = pad_collate(items_annotated)
        
        batch_x = batch[0].to(model.device)
        batch_y = batch[1].to(model.device)
        batch_w = batch[2].to(model.device)
        
        # Forward pass
        y_pred = model_ft(batch_x)
    
        # Loss
        loss = loss_fn(y_pred, batch_y, batch_w)
    
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        if n_valid > 0:
            valid_dice = evaluate_stack(
                    model_ft, inputs[len(annotations) - n_valid:],
                    annotations[len(annotations) - n_valid:], batch_size, 
                    metrics=metrics, transform=eval_transform)["dice"]
            if best_dice < valid_dice:
                best_iter = i
                best_dice = valid_dice
                best_state_dict = model_ft.state_dict()
        else:
            valid_dice = 0.0
        
        if verbose and n_iter >= 10 and (i + 1) % (n_iter // 10) == 0:
            print("{}: dice = {:.6f} - val_dice = {:.6f}".format(
                i + 1,
                evaluate_stack(model_ft, inputs[:len(annotations) - n_valid], 
                               annotations[:len(annotations) - n_valid], batch_size, 
                               metrics=metrics, transform=eval_transform)["dice"],
                valid_dice))
                
    # Load best model found
    if n_valid > 0:
        if verbose:
            print("Best model fine tuned in iteration %d." % best_iter)
        model_ft.load_state_dict(best_state_dict)
    
    # Set the model to evaluation mode and return it
    model_ft.eval()
    return model_ft