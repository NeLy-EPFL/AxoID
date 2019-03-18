#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains code useful for using internal model for tracking axon identities.
Created on Thu Mar 14 15:03:31 2019

@author: nicolas
"""
### TODOs:
#

import numpy as np
from skimage import measure
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class Axon():
    ### TODOs:
    #   1. Make shape as protected attribute ?
    """Axon object with an identity and properties linked to 2-photon imaging."""
    
    def __init__(self, identity):
        """Create and initialize an axon."""
        self.id = identity
        self.position = None
        self.area = None
        self.tdTom = None
        self.gcamp = np.array([])
        self.shape = np.array([])
        self._shape_iter = 0
    
    def init_shape(self, shape):
        """Initialize the axon's shape."""
        self.shape = shape.astype(np.float64)
        self._shape_iter = 1
    
    def update_shape(self, new_shape):
        """Update the axon shape with the new shape array."""
        # Pad the shape images to the same size
        height = max(self.shape.shape[0], new_shape.shape[0])
        width = max(self.shape.shape[1], new_shape.shape[1])
        
        shapes = np.zeros((2, height, width), np.float64)
        for i, image in enumerate([self.shape, new_shape.astype(np.float64)]):
            diff_height = height - image.shape[0]
            diff_width = width - image.shape[1]
            height_padding = (int(diff_height / 2), int(np.ceil(diff_height / 2)))
            width_padding = (int(np.floor(diff_width / 2)), int(np.ceil(diff_width / 2)))
            shapes[i] = np.pad(image, [height_padding, width_padding], 'constant')
        
        # Online average them
        self.shape = (shapes[0] * self._shape_iter + shapes[1]) / (self._shape_iter + 1)
        self._shape_iter += 1


class InternalModel():
    ### TODOs: 
    #   1. Add possibility to initialize with only segmentation (not necessarily labelled)
    #   2. Take into account shift of local CoM with more/less neurons
    #   3. Update model should add new neurons to the model (?)
    #   4. Make attributes as protected ?
    """Model of the axon structure of 2-photon images."""
    
    def __init__(self):
        """Create and initialize an internal model."""
        # Container of axons object
        self.axons = []
        # Image of the model's axons
        self.model_image = None
        # Coordinate of the center of mass
        self.center_of_mass = None
        # Iteration of update (useful for moving average)
        self._update_iter = 0
    
    def initialize(self, id_frame, id_seg):
        """(Re-)Initialize the model with the given identity frame."""
        # Re-initialize the mdoel
        self.axons = []
        self.model_image = np.zeros(id_seg.shape, np.uint8)
        self._update_iter = 1
        
        regions = measure.regionprops(id_seg, coordinates='rc')
        centroids = np.array([region.centroid for region in regions])
        areas = np.array([region.area for region in regions])[:,np.newaxis]
        self.center_of_mass = (centroids * areas).sum(0) / areas.sum()
        
        for region in regions:
            roi = id_seg == region.label
            
            # Create an axon object with a new identity
            axon = Axon(region.label)
            
            # Update its properties
            axon.position = np.array(region.centroid) - self.center_of_mass
            axon.area = region.area
            axon.tdTom = id_frame[roi, 0].mean()
            axon.gcamp = np.append(axon.gcamp, id_frame[roi, 1].mean())
            
            # Find shape array around centroid
            local_h, local_w = region.image.shape
            r_s = max(region.local_centroid[0], local_h - region.local_centroid[0])
            c_s = max(region.local_centroid[1], local_w - region.local_centroid[1])
            axon.init_shape(np.pad(region.image, [(int(r_s - region.local_centroid[0] + 0.5), 
                                                   int(r_s - (local_h - region.local_centroid[0]) + 0.5)),
                                                  (int(c_s - region.local_centroid[1] + 0.5), 
                                                   int(c_s - (local_w - region.local_centroid[1]) + 0.5))],
                                   'constant'))
            
            self.axons.append(axon)
        # Draw the model
        self._draw_model()
    
    def match_frame(self, frame, seg):
        """Match neurons of the given frame to the model's."""
        # Extract ROIs and compute local center of mass
        labels = measure.label(seg, connectivity=1)
        regions = measure.regionprops(labels, coordinates='rc')
        centroids = np.array([region.centroid for region in regions])
        areas = np.array([region.area for region in regions])[:, np.newaxis]
        local_CoM = (centroids * areas).sum(0) / areas.sum()
        
        # Build a cost matrix where rows are new ROIs and cols are model's axons
        # Currently, cost = distance
        x_roi = centroids - local_CoM
        x_model = np.stack([axon.position for axon in self.axons], axis=0)
        cost_matrix = cdist(x_roi, x_model)
        
        # Assign identities through the hungarian method
        rows_ids, col_ids = linear_sum_assignment(cost_matrix)
        ids = dict(zip([regions[i].label for i in rows_ids],
                       [self.axons[i].id for i in col_ids]))
        
        # Make identity image, and return it
        identities = np.zeros(seg.shape, np.uint8)
        for region in [regions[i] for i in rows_ids]:
            roi = labels == region.label
            identities[roi] = ids[region.label]
        return identities
    
    def update_model(self, id_frame, id_seg):
        """Update the model based on the identity frame."""
        # Axons' identity to index mapping
        id_to_idx = dict()
        for i in range(len(self.axons)):
            id_to_idx.update({self.axons[i].id: i})
        
        regions = measure.regionprops(id_seg, coordinates='rc')
        centroids = np.array([region.centroid for region in regions])
        areas = np.array([region.area for region in regions])[:, np.newaxis]
        local_CoM = (centroids * areas).sum(0) / areas.sum()
        
        for region in regions:
            idx = id_to_idx[region.label]
            roi = id_seg == region.label
            
            position = np.array(region.centroid) - local_CoM
            self.axons[idx].position = self._update_mean(self.axons[idx].position, position)
            self.axons[idx].area = self._update_mean(self.axons[idx].area, region.area)
            self.axons[idx].tdTom = self._update_mean(self.axons[idx].tdTom, id_frame[roi, 0].mean())
            self.axons[idx].gcamp = np.append(self.axons[idx].gcamp, id_frame[roi, 1].mean())
            
            # Find shape array around centroid and update it
            local_h, local_w = region.image.shape
            r_s = max(region.local_centroid[0], local_h - region.local_centroid[0])
            c_s = max(region.local_centroid[1], local_w - region.local_centroid[1])
            new_shape = np.pad(region.image, [(int(r_s - region.local_centroid[0] + 0.5), 
                                               int(r_s - (local_h - region.local_centroid[0]) + 0.5)),
                                              (int(c_s - region.local_centroid[1] + 0.5), 
                                               int(c_s - (local_w - region.local_centroid[1]) + 0.5))],
                                'constant')
            self.axons[idx].update_shape(new_shape)
        
        self.center_of_mass = self._update_mean(self.center_of_mass, local_CoM)
        self._update_iter += 1
        # Update the model image
        self._draw_model()
    
    def _update_mean(self, mean, new_value):
        """Update the online mean with the new value."""
        new_mean = (mean * self._update_iter + new_value) / (self._update_iter + 1)
        return new_mean
    
    def _draw_model(self):
        """Update the model image based on the axons' properties."""
        model_image = np.zeros_like(self.model_image)
        
        for axon in self.axons:
            # Create binary axon segmentation from axon shape and area
            axon_seg = axon.shape >= np.sort(axon.shape, axis=None)[- int(axon.area + 0.5)]
            
            # Center of the axon on the image
            center = (axon.position + self.center_of_mass).astype(np.uint16)
            # Compute row and col coordinates to fit the axon in the image
            min_row = max(0, axon_seg.shape[0] // 2 - center[0])
            min_col = max(0, axon_seg.shape[1] // 2 - center[1])
            max_row = min(axon_seg.shape[0] // 2, model_image.shape[0] - center[0]) + axon_seg.shape[0] // 2
            max_col = min(axon_seg.shape[1] // 2, model_image.shape[1] - center[1]) + axon_seg.shape[1] // 2
            axon_seg = axon_seg [min_row:max_row, min_col:max_col]
            
            # Compute the axon position in the image
            min_row = max(0, center[0] - axon_seg.shape[0] // 2)
            min_col = max(0, center[1] - axon_seg.shape[1] // 2)
            max_row = min(model_image.shape[0], center[0] + axon_seg.shape[0] // 2)
            max_col = min(model_image.shape[1], center[1] + axon_seg.shape[1] // 2)
            model_image[min_row:max_row, min_col:max_col] = axon_seg * axon.id
            
        self.model_image = model_image