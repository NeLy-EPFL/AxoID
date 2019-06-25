#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for testing models on data with PyTorch.
Created on Thu Oct 25 16:09:50 2018

@author: nicolas
"""

import numpy as np
import matplotlib.pyplot as plt

import torch, torchvision

from axoid.utils.image import imread_to_float, overlay_preds_targets


def predict(model, dataloader, discard_target=True):
    """
    Return predictions for the given dataloader and model.
    
    Parameters
    ----------
    model : pytorch model
        The model which will be used to make the predictions.
    dataloader : pytorch dataloader
        Dataloader returning batches of inputs.
    discard_target : bool (default = True)
        If True, onyl the first element of the batch is kept. This is useful if
        the dataloader returns batches as (inputs, targets, ...) tuples.
    
    Returns
    -------
    predictions : tensor
        Tensor (on cpu) of raw predictions. It is the user's responsability 
        to convert the raw predictions if necessary (e.g. logits to proba).
    """
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
    return predictions.cpu()
    
def predict_stack(model, stack, batch_size, input_channels="RG", 
                  numpy_channels_last=True, transform=None):
    """
    Output predictions for the given image stack and model.
    
    Parameters
    ----------
    model : pytorch model
        The model which will be used to make the predictions.
    stack : str or ndarray or tensor
        Input stack of images to predict.
        If string, it is the filename of the stack.
        If ndarray, it is the stack of images array.
        If tensor, it is considered already with correct shape (channels first,
        and correct input_channels).
    batch_size : int
        Number of images to send to the network at once.
    input_channels : str (default = "RG")
        Input channels to use (Red Green Blue). Is ignored if stack is a tensor.
    numpy_channels_last : bool (default = True)
        If True, and stack is an ndarray, it will be converted from NHWC to NCHW.
    transform : callable (optional)
        Input transformation function.
        /!\ Takes a ndarray as input. (This is to be consistent with other functions).
    
    Returns
    -------
    predictions : tensor
        Tensor (on cpu) of raw predictions. It is the user's responsability 
        to convert the raw predictions if necessary (e.g. logits to proba).
    """
    # Make sure stack is in the correct shape
    if isinstance(stack, str):
        stack = imread_to_float(stack, scaling=255)
        channels = {"R": stack[...,0], "G": stack[...,1], "B": stack[...,2]}
        stack = np.stack([channels[c] for c in input_channels], axis=1)
        stack = torch.from_numpy(stack)
    elif isinstance(stack, np.ndarray):
        if numpy_channels_last: # change to channels first
            channels = {"R": stack[...,0], "G": stack[...,1], "B": stack[...,2]}
            stack = np.stack([channels[c] for c in input_channels], axis=1)
        stack = torch.from_numpy(stack)
    elif isinstance(stack, torch.Tensor):
        pass
    else:
        raise TypeError("Unknown type %s for the image stack." % type(stack))
    
    # Apply transform if applicable
    if transform is not None:
        stack = torch.from_numpy(transform(stack.numpy()))
    
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
    return predictions.cpu()


def evaluate(model, dataloader, metrics):
    """
    Return the average metric values for the given dataloader and model.
    
    Parameters
    ----------
    model : pytorch model
        The model which will be used to make the predictions.
    dataloader : pytorch dataloader
        Dataloader returning batches of inputs.
    metrics : dict
        Dictionary where keys are metric names (str), and values are metric 
        function (callable taking predictions, targets as inputs).
    
    Returns
    -------
    values : dict
        Dictionary where keys are metric names (str), and values are metric 
        average values.
    """
    values = {}
    for key in metrics.keys():
        values[key] = 0
    
    # Compute metrics over all data
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch_x = batch[0].to(model.device)
            batch_y = batch[1].to(model.device)
            
            y_pred = model(batch_x)

            for key in metrics.keys():
                values[key] += metrics[key](y_pred, batch_y).item() * batch_x.shape[0]
            
    for key in values.keys():
        values[key] /= len(dataloader.dataset)
    return values

def evaluate_stack(model, input_stack, target_stack, batch_size, metrics,
                   input_channels="RG", numpy_channels_last=True, transform=None):
    """
    Return the average metric values for the given stack and model.
    
    Parameters
    ----------
    model : pytorch model
        The model which will be used to make the predictions.
    input_stack : str or ndarray or tensor
        Input stack of images to predict.
        If string, it is the filename of the stack.
        If ndarray, it is the stack of images array.
        If tensor, it is considered already with correct shape (channels first,
        and correct input_channels).
    target_stack : str or ndarray or tensor
        Input stack of images to predict.
        If string, it is the filename of the stack.
        If ndarray, it is the stack of images array.
        If tensor, it is the stack of images tensor.
    batch_size : int
        Number of images to send to the network at once.
    metrics : dict
        Dictionary where keys are metric names (str), and values are metric 
        function (callable taking predictions, targets as inputs).
    input_channels : str (default = "RG")
        Input channels to use (Red Green Blue). Is ignored if input_stack is a tensor.
    numpy_channels_last : bool (default = True)
        If True, and input_stack is an ndarray, it will be converted from NHWC to NCHW.
    transform : callable (optional)
        Input transformation function.
        /!\ Takes a ndarray as input. (This is to be consistent with other functions).
    
    Returns
    -------
    values : dict
        Dictionary where keys are metric names (str), and values are metric 
        average values.
    """
    # Make sure target_stack is in the correct shape
    if isinstance(target_stack, str):
        target_stack = imread_to_float(target_stack, scaling=255)
        target_stack = torch.from_numpy(target_stack)
    elif isinstance(target_stack, np.ndarray):
        target_stack = torch.from_numpy(target_stack)
    elif isinstance(target_stack, torch.Tensor):
        pass
    else:
        raise TypeError("Unknown type %s for the image stack." % type(target_stack))
        
    # Make predictions
    predictions = predict_stack(model, input_stack, batch_size, 
                                input_channels=input_channels, 
                                numpy_channels_last=numpy_channels_last, 
                                transform=transform)
    
    # Compute metrics
    values = {}
    for key in metrics.keys():
        values[key] = metrics[key](predictions, target_stack).item()
    return values


def _make_images_valid(images):
    """Make sure the given images have correct value range and number of channels."""
    # Set range from [min,max] to [0,1]
    images = (images - images.min()) / (images.max() - images.min())
    # If only 2 channel (e.g. "RG"), add a third one which is a copy of the second (so B = G)
    if images.shape[1] == 2:
        green = images[:,1,:,:].unsqueeze(1)
        images = torch.cat([images, green], 1)
    return images

def show_sample(model, dataloader, n_samples=4, metrics=None):
    """
    Display random samples of inputs, predictions, and targets.
    
    Parameters
    ----------
    model : PyTorch model
        The model, based on Torch.nn.Module. It should have a `device` 
        attribute.
    dataloader : PyTorch DataLoader
        The data will be sampled from the DataLoader's dataset.
    n_samples : int (default = 4)
        Number of images in the random sampling.
    metrics : dict of {str : callable}
        Dictionary of metrics to be computed over the samples. It should 
        take 3 tensors as input (predictions, targets, and masks), and 
        output a scalar tensor.
    """
    indices = np.random.randint(0, len(dataloader.dataset), n_samples)
    items = [dataloader.dataset[i] for i in indices]
    items = dataloader.collate_fn(items)
    
    inputs = items[0].to(model.device)
    targets = items[1].to(model.device)
    if len(items) == 3: # loss weights
        weights = items[2].to(model.device)
    
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
    
    preds = torch.sigmoid(preds) > 0.5
        
    # Modify inputs to make sure it is a valid image
    inputs = _make_images_valid(inputs)
    
    height, width = inputs.shape[-2:]
    outs = torchvision.utils.make_grid(inputs, pad_value=1.0)
    outs_p = torchvision.utils.make_grid(preds.view([-1, 1, height, width]), pad_value=1.0)
    outs_t = torchvision.utils.make_grid(targets.view([-1, 1, height, width]), pad_value=0)
    if len(items) == 3:
        outs_w = torchvision.utils.make_grid(weights.view([-1, 1, height, width]), pad_value=1.0)
    
    if len(items) == 3:
        plt.figure(figsize=(13,10))
        plt.subplot(311); plt.title("Inputs")
        plt.imshow(outs.cpu().numpy().transpose([1,2,0]), vmin=0, vmax=1)
        plt.subplot(312); plt.title("Predictions and ground truths")
        plt.imshow(overlay_preds_targets(
                outs_p.cpu().numpy().clip(0,1)[0],
                outs_t.cpu().numpy()[0]))
        plt.subplot(313); plt.title("Pixel-wise loss weights")
        plt.imshow(outs_w.cpu().numpy()[0], vmin=0, vmax=1, cmap="gray")
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(13,7))
        plt.subplot(211); plt.title("Inputs")
        plt.imshow(outs.cpu().numpy().transpose([1,2,0]), vmin=0, vmax=1)
        plt.subplot(212); plt.title("Predictions and ground truths")
        plt.imshow(overlay_preds_targets(
                outs_p.cpu().numpy().clip(0,1)[0],
                outs_t.cpu().numpy()[0]))
        plt.tight_layout()
        plt.show()
