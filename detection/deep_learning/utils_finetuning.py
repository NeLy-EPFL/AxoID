#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for fine tuning with PyTorch.
Created on Mon Apr  8 14:13:08 2019

@author: nicolas
"""

import time, copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from skimage import measure, draw

import torch

from utils_data import pad_collate, normalize_range, pad_transform_stack, compute_weights
from utils_loss import get_BCEWithLogits_loss
from utils_metric import get_dice_metric
from utils_test import evaluate_stack


def fine_tune(model, inputs, annotations, weights=None, n_iter=200, n_valid=1,
              batch_size=16, learning_rate = 0.0005, verbose=1):
    """Fine tune the given model on the annotated data, and return the resulting model."""
    u_depth = len(model.convs)
    device = model.device
    annotated_per_batch = min(len(annotations) - n_valid, batch_size) 
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
        best_state_dict = copy.deepcopy(model_ft.state_dict())
    
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
                best_state_dict = copy.deepcopy(model_ft.state_dict())
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


class ROISelector(widgets.LassoSelector):
    """Use matplotlib to draw ROIs for the given images."""
    
    def __init__(self, images):
        """Create the figure and start the selection with the first frame."""
        # If single image (RGB), change it to a stack of a single image
        if images.ndim == 3:
            self.images = images.copy()[np.newaxis, ...]
        else:
            self.images = images.copy()
        self.index = 0
        
        self.fig = plt.figure(figsize=(9,5))
        self.fig.suptitle("\n".join(["ROI Selector", 
                                     "Press backspace to delete last selection", 
                                     "Press enter to validate selected ROIs"]))
        
        # Set first plot: selection
        self.ax = self.fig.add_subplot(121)
        self.title = ["Frame 1/%d" % len(self.images), "0 ROI"]
        
        # Set the second plot: segmentation results
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_title("Resulting segmentation")
        self.segmentation = np.zeros(self.images.shape[:-1], np.bool)
        self.polygons = []
        
        self.update()
        
        # Initialize the LassoSelector
        super().__init__(self.ax, onselect=self.onselect)
        
        self.cid_key_press = self.canvas.mpl_connect("key_press_event", self.key_press)
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.83])
        self.fig.show()

    def update(self):
        """Update the display."""
        # Change title to display number of ROIs
        self.title[0] = "Frame %d/%d" % (self.index + 1, len(self.images))
        _, n_roi = measure.label(self.segmentation[self.index], connectivity=1, return_num=True)
        self.title[1] = "%d ROI" % n_roi + ("s" if n_roi > 1 else "")
        self.ax.set_title("\n".join(self.title))
        self.ax.imshow(self.images[self.index])
        
        # Draw images
        self.ax2.imshow(self.segmentation[self.index], cmap="gray")
        self.fig.canvas.draw_idle()
    
    def onselect(self, verts):
        """Called when the lasso selector is released."""
        vertices = np.array(verts)
        
        # Draw the ROI and keep track of the polygons
        polygon, = self.ax.fill(vertices[:, 0], vertices[:, 1], "w", alpha=0.5)
        self.polygons.append(polygon)
        rr, cc = draw.polygon(vertices[:, 1], vertices[:, 0],
                              shape=self.segmentation[self.index].shape)
        self.segmentation[self.index, rr, cc] = True
        
        self.update()
    
    def disconnect(self):
        """Stop the ROI selection."""
        self.disconnect_events()
        self.canvas.mpl_disconnect(self.cid_key_press)
    
    def key_press(self, event):
        """Callback for key press events."""
        if event.key == "enter":
            # Go to next image, or finish
            self.index += 1
            if self.index >= len(self.images):
                self.fig.suptitle("ROI Selector\nROIs validated")
                self.fid.canvas.draw_idle()
                self.disconnect()
            else:
                self.polygons = []
                self.ax.clear()
                self.update()
            
        elif event.key == "backspace" and len(self.polygons) > 0:
            # Erase last drawn ROI
            polygon = self.polygons.pop()
            vertices = polygon.get_xy()
            rr, cc = draw.polygon(vertices[:, 1], vertices[:, 0],
                                  shape=self.segmentation[self.index].shape)
            self.segmentation[self.index, rr, cc] = False
            polygon.remove()
            
            self.update()