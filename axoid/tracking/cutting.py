#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module with function useful for "cuts" of ROIs. 
Cuts are a linear separation of an ROI into 2 ROIs, effectively making 2 axons 
out of it.
/!\ It assumes ROIs elongated vertically. If they are horizontal, the code is 
unable to guess which is the same side (it simply takes the top elongated side
as common, whatever its orientation).
Created on Tue Jun 11 14:51:46 2019

@author: nicolas
"""

from copy import copy

import numpy as np
from skimage import measure
import cv2

from axoid.detection.clustering import segment_projection


def fit_line(image):
    """
    Fit a line to the binary object in image, and return its parameters.
    
    Parameters
    ----------
    image : ndarray
        Binary image where foreground pixels will be used to fit the line.
    
    Returns
    -------
    n : ndarray
        Normal unit vector to the line. It is in the row-col coordinate systems,
        as opposed to x-y (inverted).
    d : float
        Shortest distance from the line to the origin (top-left corner).
    """
    # Coordinates of pixels
    coords = np.array(np.where(image)).T
    
    # Fit a line model
    lm = measure.LineModelND()
    lm.estimate(coords)
    origin, direction = lm.params
    
    # Parameters
    n = np.array([-direction[1], direction[0]])
    n /= np.linalg.norm(n)
    d = np.dot(origin, n)
    
    # Assures a positive distance
    if d < 0:
        d *= -1
        n *= -1
    
    return n, d


def norm_to_ellipse(n, d, roi_img):
    """
    Normalize the line parameters to an ellipse fitted on the ROI.
    
    Parameters
    ----------
    n : ndarray
        Normal unit vector to the line. It is in the row-col coordinate systems,
        as opposed to x-y (inverted).
    d : float
        Shortest distance from the line to the origin (top-left corner).
    roi_img : ndarray
        Binary image with a single ROI.
    
    Returns
    -------
    n : ndarray
        Normal unit vector to the line, normalized to the ROI ellipse.
    d : float
        Shortest distance from the line to the origin, normalized to the ROI ellipse.
    """
    new_n = n.copy()
    new_d = copy(d)
    
    # Fit an ellipse to the ROI
    _, contours, _ = cv2.findContours(roi_img.astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipse = cv2.fitEllipse(contours[0])
    center = np.array([ellipse[0][1], ellipse[0][0]])
    axes = np.array([ellipse[1][1], ellipse[1][0]]) / 2
    rotation = - ellipse[2] * np.pi / 180
    if -np.pi < rotation < -np.pi/2: # correct angle
        rotation += np.pi
    
    # Normalize line parameters w.r.t. the ellipse
    # Centering
    new_d -= np.dot(center, new_n)
    # Rotation (note that we take the negative angle)
    rot_matrix = np.array([[np.cos(rotation), np.sin(rotation)], 
                           [-np.sin(rotation), np.cos(rotation)]])
    new_n = rot_matrix @ new_n
    # Scaling
    new_n /= axes[::-1]
    new_d /= axes[0] * axes[1]
    new_d /= np.linalg.norm(new_n)
    new_n /= np.linalg.norm(new_n)
    
    return new_n, new_d

def fit_to_ellipse(n, d, roi_img):
    """
    Transform the normalized line parameters to an ellipse fitted on the ROI.
    
    Parameters
    ----------
    n : ndarray
        Normal unit vector to the line, normalized to an ellipse.
    d : float
        Shortest distance from the line to the origin, normalized to an ellipse.
    roi_img : ndarray
        Binary image with a single ROI.
    
    Returns
    -------
    n : ndarray
        Normal unit vector to the line, fitted to the ROI ellipse.
    d : float
        Shortest distance from the line to the origin, fitted to the ROI ellipse.
    """
    new_n = n.copy()
    new_d = copy(d)
    
    # Fit an ellipse to the ROI
    _, contours, _ = cv2.findContours(roi_img.astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ellipse = cv2.fitEllipse(contours[0])
    center = np.array([ellipse[0][1], ellipse[0][0]])
    axes = np.array([ellipse[1][1], ellipse[1][0]]) / 2
    rotation =  - ellipse[2] * np.pi / 180
    if -np.pi < rotation < -np.pi/2: # correct angle
        rotation += np.pi
    
    # Transform line parameters w.r.t. the ellipse
    # Scaling
    new_n *= axes[::-1]
    new_d *= axes[0] * axes[1]
    new_d /= np.linalg.norm(new_n)
    new_n /= np.linalg.norm(new_n)
    # Rotation
    rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation)], 
                           [np.sin(rotation), np.cos(rotation)]])
    new_n = rot_matrix @ new_n
    # Centering
    new_d += np.dot(center, new_n)
    
    return new_n, new_d


def find_cuts(projection, model, min_area=None):
    """
    Return a dictionary mapping axon identity to a list of cuts (as parametrized lines).
    
    This is to be used in the AxoID pipeline, where it assumes that there was
    fine tuning, using similar frames. Projection is the temporal mean of these
    frames (with potential smoothing if there were not many frames). Thus,
    projection was used as the initialization frame for the tracker model.
    
    Parameters
    ----------
    projection : ndarray
        Temporal projection of the similar frames used for fine tuning, and to
        initialize the tracker model.
    model : InternalModel
        Internal model of the tracker.
    min_area : int (optional)
        Minimum size of ROIs in pixels, under which ROIs are discarded.
    
    Returns
    -------
    cuts : dict
        Dictionary mapping axon identities (keys) to a list of cuts (values).
        Each element of the list corresponds to a cut, i.e. a tuple of line 
        parameters (n, d). The list is sorted by ascending distance d.
    """
    # Find cuts as separation of ROIs in segmentation
    segmentation = segment_projection(projection, min_area)
    separations = np.logical_and(segmentation, np.logical_not(
            segment_projection(projection, min_area, separation_border=True)))
    
    # Link cuts identity to the ROIs
    separations = separations.astype(model.image.dtype) * model.match_frame(
            projection, segmentation)
    
    # Parametrize a line for each cuts (normal vector + distance to origin)
    cuts = dict([(axon.id, []) for axon in model.axons])
    for region in measure.regionprops(separations):
        n, d = fit_line(separations == region.label)
        
        # Normalize the cut to an ellipse fitting of the ROI
        n, d = norm_to_ellipse(n, d, model.image == region.label)
        
        cuts[region.label].append((n, d))
    
    # Sort the cuts by increasing distance d (useful for applying cuts)
    for axon in model.axons:
        cuts[axon.id] = sorted(cuts[axon.id], key=lambda line: line[1])
    
    return cuts


def get_cut_pixels(n, d, roi_img):
    """
    Return the coordinates of the pixels on the positive side of the cut.
    
    The "positive side" is where coords @ n > d, i.e. the side further away from
    the ellipse's center.
    
    Parameters
    ----------
    n : ndarray
        Normal unit vector to the line, normalized to an ellipse.
    d : float
        Shortest distance from the line to the origin, normalized to an ellipse.
    roi_img : ndarray
        Binary image with a single ROI.
    
    Returns
    -------
    coords : ndarray
        Coordinate array where each row is a pixel coordinate in row-col, of
        every pixel on the "positive side" of the cut.
    """
    # Fit the line to the ROI
    n, d = fit_to_ellipse(n, d, roi_img)
    
    # Find pixels on the positive sides of the cut
    coords = np.array(np.where(roi_img)).T
    y = coords @ n > d
    return coords[y].astype(np.uint16)

def apply_cuts(cuts, model, identities=None):
    """
    Apply the cuts to the ROIs of the model image and all identity frames.
    
    Parameters
    ----------
    cuts : dict
        Dictionary mapping axon identities (keys) to a list of cuts (values).
        Each element of the list corresponds to a cut, i.e. a tuple of line 
        parameters (n, d). The list have to be sorted by ascending distance d.
    model : InternalModel
        Internal model of the tracker.
    identities : ndarray (optional)
        Stack of label images corresponding to tracked ROI identities.
    
    Returns
    -------
    new_model_image : ndarray
        Image of the model after applying the cuts. Note that model is not modified.
    new_identities : ndarray (optional)
        Stack of label images corresponding to tracked ROI identities after the cuts.
    """
    new_model_image = model.image.copy()
    if identities is not None:
        new_identities = identities.copy()
    
    # Create new ids for ROIs created after the cut
    new_ids = dict([(axon.id, []) for axon in model.axons])
    id_counter = max([axon.id for axon in model.axons])
    for axon in model.axons:
        for _ in cuts[axon.id]:
            id_counter += 1
            new_ids[axon.id].append(id_counter)
    
    # Loop over ROIs
    for axon in model.axons:
        # Only consider axons with cuts
        if len(cuts[axon.id]) == 0:
            continue
        
        # Cut the model image
        roi_img = model.image == axon.id
        for j, (n, d) in enumerate(cuts[axon.id]):
            new_coords = get_cut_pixels(n, d, roi_img)
            new_model_image[new_coords[:,0], new_coords[:,1]] = new_ids[axon.id][j]
        
        # Cut the identity frames
        if identities is not None:
            for k in range(len(identities)):
                roi_img = identities[k] == axon.id
                if np.sum(roi_img) == 0:
                    continue            
                for j, (n, d) in enumerate(cuts[axon.id]):
                    new_coords = get_cut_pixels(n, d, roi_img)
                    new_identities[k, new_coords[:,0], new_coords[:,1]] = new_ids[axon.id][j]
    
    if identities is None:
        return new_model_image
    else:
        return new_model_image, new_identities