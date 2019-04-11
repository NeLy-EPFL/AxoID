#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for fine tuning with PyTorch.
Created on Mon Apr  8 14:13:08 2019

@author: nicolas
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from skimage import measure, draw
import cv2

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


class ROIAnnotator_mpl(widgets.LassoSelector):
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


class ROIAnnotator():
    """Use OpenCV to draw ROI for the given frames."""
    
    def __init__(self, images, window_name="ROI Annotation", rgb2bgr=True):
        """
        Start the ROI annotation process with the given images.
        
        Args:
            images: ndarray (N, H, W, C)
                Numpy array with the input images on which to annotate the ROIs
            rgb2bgr: bool (default: True)
                If True, `images` will be converted from RGB to BGR (OpenCV works
                with BGR).
        """
        # Initialization        
        self.images = images.copy()
        # Change from RGB to BGR (for OpenCV compatibility)
        if rgb2bgr:
            self.images = self.images[..., ::-1]
        # Make sure images is float32
        if self.images.dtype == np.uint8:
            self.images = (self.images / 255).astype(np.float32)
        elif self.images.dtype == np.float64:
            self.images = self.images.astype(np.float32)
        
        self.window = window_name
        self.tb_brush_size = "Brush size"
        self.tb_alpha = "ROI opacity"
        
        self.idx = 0 # index of the current image to annotated
        self.segmentations = np.zeros(self.images.shape[:-1], np.bool)
        # Lists of arrays with the brushstrokes images (useful for undo/redo)
        self.brushstrokes = []
        for _ in range(len(self.images)):
            self.brushstrokes.append([])
        self.drawing = False
        self.x, self.y = -1, -1
        self.cursor = np.zeros(self.images.shape[1:], self.images.dtype) # cursor image
        self.border_margin = 5 # number of pixels between overlay and segmentation
        self.info_height = 50 # height of the bottom information rectangle
        
        # Main function (create and destroy window, deal with loop and waitKey, etc.)
        self.main()
    
    def imshow(self):
        """Draw the current image."""
        alpha = cv2.getTrackbarPos(self.tb_alpha, self.window) / 10
        image = self.images[self.idx]
        # Create the segmentation image
        self.segmentations[self.idx] = np.max([np.zeros(image.shape[:-1])] + \
                                               self.brushstrokes[self.idx], axis=0)
        
        # Create the overlay of image and drawn ROIs
        if len(self.brushstrokes[self.idx]) > 0:
            rois = cv2.cvtColor(self.segmentations[self.idx].astype(image.dtype), 
                                cv2.COLOR_GRAY2BGR)
            overlay = (1 - alpha * rois) * image + alpha * rois
        else:
            overlay = image.copy()
        # Create an overlay with the cursor
        overlay = (1 - alpha * self.cursor) * overlay + alpha * self.cursor
        
        # Concatenate with the current segmentation (with a white border)
        seg = cv2.cvtColor(self.segmentations[self.idx].astype(overlay.dtype), 
                           cv2.COLOR_GRAY2BGR)
        img_display = np.concatenate(
                [overlay, np.ones((image.shape[0], self.border_margin, 3)), seg], axis=1)
        
        # Concatenate this image with a rectangle containing informations
        info_rect = np.ones((self.info_height, img_display.shape[1], 3))
        _, n_roi = measure.label(self.segmentations[self.idx], connectivity=1, return_num=True)
        cv2.putText(info_rect, 
                    "Frame %d/%d" % (self.idx + 1, len(self.images)), 
                    (image.shape[1] // 2 - 45, self.info_height // 2 - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(info_rect, 
                    "%d ROI" % n_roi + ("s" if n_roi > 1 else ""), 
                    (image.shape[1] // 2 - 25, self.info_height // 2 + 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(info_rect, 
                    "Resulting segmentation", 
                    (image.shape[1] + self.border_margin + image.shape[1] // 2 - 95,
                     self.info_height // 2 + 5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
        final_img = np.concatenate((img_display, info_rect))
        
        # Make display image uint8 to have x, y, R, G, B infos in the window
        final_img = (final_img * 255).astype(np.uint8)
        
        cv2.imshow(self.window, final_img)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for the OpenCV window."""
        # Start drawing if on the image
        if event == cv2.EVENT_LBUTTONDOWN and \
           x < self.images.shape[2] and y < self.images.shape[1]:
            brush_size = cv2.getTrackbarPos(self.tb_brush_size, self.window)
            self.drawing = True
            self.x, self.y = x, y
            self.brushstrokes[self.idx].append(np.zeros(self.images.shape[1:-1]))
            cv2.line(self.brushstrokes[self.idx][-1], (x, y), (self.x, self.y), 1.0, brush_size)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            brush_size = cv2.getTrackbarPos(self.tb_brush_size, self.window)
            # Draw the cursor under the mouse
            self.cursor = np.zeros(self.images.shape[1:], self.images.dtype)
            cv2.line(self.cursor, (x, y), (x, y), (1.0, 1.0, 1.0), brush_size)
            
            # Continue the brush stroke
            if self.drawing == True:
                cv2.line(self.brushstrokes[self.idx][-1], (x, y), (self.x, self.y), 1.0, brush_size)
                self.x, self.y = x, y
                
        # Terminate the brush stroke
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    
    def key_callback(self, key):
        """Key callback for the OpenCV window."""
        terminate = False
        
        # Escape: close the window and stop the annotation
        if key == 27:
            print("ESC pressed, annotation interrupted.")
            terminate = True
            
        # Backspace: erase last brushstroke
        elif key == ord('\b'):
            if len(self.brushstrokes[self.idx]) > 0:
                self.brushstrokes[self.idx].pop()
                
        # Enter: validate current ROIs and go to the next image
        elif key == ord('\r') or key == ord('\n'):
            self.idx += 1
            if self.idx >= len(self.images):
                print("Annotation finished.")
                terminate = True
        
        # Navigate images with numbers or right/left arrows
        elif ord('1') <= key <= ord(str(len(self.images))):
            self.idx = int(chr(key)) - 1
        elif key == 81: # left arrow
            self.idx = max(0, self.idx - 1)
        elif key == 83: # right arrow
            self.idx = min(len(self.images) - 1, self.idx + 1)
                
        return terminate
    
    def main(self):
        """Main loop of the ROI annotation."""
        # Create the window
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 2 * (self.images.shape[2] * 2 + self.border_margin),
                         2 * self.images.shape[1])
        cv2.setMouseCallback(self.window, self.mouse_callback)
        
        # Create the trackbars
        cv2.createTrackbar(self.tb_brush_size, self.window, 5, 10, lambda x: None)
        cv2.setTrackbarMin(self.tb_brush_size, self.window, 1)
        cv2.createTrackbar(self.tb_alpha, self.window, 5, 10, lambda x: None)
        
        # Main loop
        terminate = False
        while not terminate:
            self.imshow()
            key = cv2.waitKey(33)
            if key != -1: # a key is pressed
                key &= 0xFF # keep only last byte
                terminate = self.key_callback(key)
        
        # Terminate the annotation
        cv2.destroyWindow(self.window)