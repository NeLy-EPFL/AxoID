#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for testing with PyTorch.
Created on Thu Oct 25 16:09:50 2018

@author: nicolas
"""

import numpy as np
import matplotlib.pyplot as plt

import torch, torchvision

from utils_data import make_images_valid
from utils_common.image import imread_to_float, overlay_preds_targets


def predict(model, dataloader, discard_target=True):
    """Output predictions for the given dataloader and model.
    
    `discard_target` can be used if the dataloader return batches as 
    (inputs, targets, ...) tuples."""
    predictions = []
    
    # Compute predictions
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if discard_target:
                batch = batch[0]
                
            batch = batch.to(model.device)
            predictions.append(model(batch))
    
    # Concatenate everything together
    predictions = torch.cat(predictions)
    return predictions
    
def predict_stack(model, stack, batch_size, input_channels="RG"):
    """Output predictions for the given image stack and model.
    
    `stack` can either be the filename (`input_channels` is then required),
    or an ndarray/tensor."""
    # Make sure stack is in the correct shape
    if isinstance(stack, str):
        stack = imread_to_float(stack, scaling=255)
        channels = {"R": stack[...,0], "G": stack[...,1], "B": stack[...,2]}
        stack = np.stack([channels[c] for c in input_channels], axis=1)
    elif isinstance(stack, np.ndarray):
        stack = torch.from_numpy(stack)
    elif isinstance(stack, torch.Tensor):
        pass
    else:
        raise TypeError("Unknown type %s for the image stack." % type(stack))
    
    predictions = []
    
    # Compute predictions
    model.eval()
    with torch.no_grad():
        for i in range(int(np.ceil(len(stack) / batch_size))):
            batch = stack[i * batch_size: (i + 1) * batch_size]
            batch = batch.to(model.device)
            predictions.append(model(batch))
    
    # Concatenate everything together
    predictions = torch.cat(predictions)
    return predictions


def evaluate(model, dataloader, metrics):
    """Return the metric values for the given dataloader and model."""
    values = {}
    for key in metrics.keys():
        values[key] = 0
    
    # Compute metrics over all data
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(model.device)
            batch_y = batch_y.to(model.device)
            
            y_pred = model(batch_x)

            for key in metrics.keys():
                values[key] += metrics[key](y_pred, batch_y).item() * batch_x.shape[0]
            
    for key in values.keys():
        values[key] /= len(dataloader.dataset)
    return values


def show_sample(model, dataloader, n_samples=4, metrics=None):
    """
    Display a random sample of some inputs, predictions, and targets.
    
    Args:
        model: PyTorch model
            The model, based on Torch.nn.Module. It should have a `device` 
            attribute.
        dataloader: PyTorch DataLoader
            The data will be sampled from the DataLoader's dataset.
        n_samples: int (default = 4)
            Number of images in the random sampling.
        metrics: dict of callable
            Dictionary of metrics to be computed over the samples. It should 
            take 3 tensors as input (predictions, targets, and masks), and 
            output a scalar tensor.
    """
    indices = np.random.randint(0, len(dataloader.dataset), n_samples)
    items = [dataloader.dataset[i] for i in indices]
    
    inputs = torch.stack([torch.from_numpy(item[0]) for item in items])
    targets = torch.stack([torch.from_numpy(item[1]) for item in items])
    inputs = inputs.to(model.device)
    targets = targets.to(model.device)
    
    with torch.no_grad():
        model.eval()
        preds = model(inputs)
        
    if metrics is not None:
        for i, idx in enumerate(indices):
            print("Image % 6d (%s): " % (idx, dataloader.dataset.x_filenames[idx]))
            for key in metrics.keys():
                print("{} = {:.6f} - ".format(key, metrics[key](preds[i].unsqueeze(0), 
                                                                targets[i].unsqueeze(0))), 
                end="")
            print("\b\b")
    
    preds = torch.tensor(torch.sigmoid(preds) > 0.5, dtype=torch.float32)
        
    # Modify inputs to make sure it is a valid image
    inputs = make_images_valid(inputs)
    
    height, width = inputs.shape[-2:]
    outs = torchvision.utils.make_grid(inputs, pad_value=1.0)
    outs_p = torchvision.utils.make_grid(preds.view([-1, 1, height, width]), pad_value=1.0)
    outs_t = torchvision.utils.make_grid(targets.view([-1, 1, height, width]), pad_value=0)
    
    plt.figure(figsize=(13,7))
    plt.subplot(211); plt.title("Inputs")
    plt.imshow(outs.cpu().numpy().transpose([1,2,0]), vmin=0, vmax=1)
    plt.subplot(212); plt.title("Predictions and ground truths")
    plt.imshow(overlay_preds_targets(
            outs_p.cpu().numpy().clip(0,1)[0],
            outs_t.cpu().numpy()[0]))
    plt.tight_layout()
    plt.show()