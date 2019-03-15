#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains code useful for using internal model for tracking axon identities.
Created on Thu Mar 14 15:03:31 2019

@author: nicolas
"""
### TODOs:
#   1. Add shape as an axon property ?

import numpy as np
from skimage import measure, draw
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class Axon():
    ### TODOs:
    #   
    """Axon object with an identity and properties linked to 2-photon imaging."""
    
    def __init__(self, identity):
        """Create and initialize an axon."""
        self.id = identity
        self.position = None
        self.area = None
        self.tdTom = None
        self.gcamp = np.array([])


class InternalModel():
    ### TODOs: 
    #   1. Simple one that does not consider (dis-)appearing neurons
    #   2. Add possibility to initialize with only segmentation (not necessarily labelled)
    #   3. Take into account shift of local CoM with more/less neurons
    #   4. Make self.model_image as a visualization for now
    """Model of the axon structure of 2-photon images."""
    
    def __init__(self):
        """Create and initialize an internal model."""
        # Container of axons object (keys are identity)
        self.axons = []
        # Image of the model's axons
        self.model_image = None
        # Coordinate of the center of mass
        self.center_of_mass = None
        # Iteration of update (useful for moving average)
        self.update_iter = 0
    
    def initialize(self, id_frame, id_seg):
        """(Re-)Initialize the model with the given identity frame."""
        # Re-initialize the mdoel
        self.axons = []
        self.model_image = np.zeros(id_seg.shape, np.uint8)
        self.update_iter = 1
        
        regions = measure.regionprops(id_seg)
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
            
            self.axons.append(axon)
        # Draw the model
        self._draw_model()
            
    def match_frame(self, frame, seg):
        """Match neurons of the given frame to the model's."""
        # Extract ROIs and compute local center of mass
        labels = measure.label(seg, connectivity=1)
        regions = measure.regionprops(labels)
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
        for region in regions:
            roi = labels == region.label
            identities[roi] = ids[region.label]
        return identities
    
    def update_model(self, id_frame, id_seg):
        """Update the model based on the identity frame."""
        # Axons' identity to index mapping
        id_to_idx = dict()
        for i in range(len(self.axons)):
            id_to_idx.update({self.axons[i].id: i})
        
        regions = measure.regionprops(id_seg)
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
            self.axons[i].gcamp = np.append(self.axons[i].gcamp, id_frame[roi, 1].mean())
        
        self.update_iter += 1
        # Update the model image
        self._draw_model()
    
    def _update_mean(self, mean, new_value):
        """Update the online mean with the new value."""
        new_mean = (mean * self.update_iter + new_value) / (self.update_iter + 1)
        return new_mean
    
    def _draw_model(self):
        """Update the model image based on the axons' properties."""
        model_image = np.zeros_like(self.model_image)
        
        for axon in self.axons:
            r, c = axon.position + self.center_of_mass
            radius = np.sqrt(axon.area / (3 * np.pi))
            
            rr, cc = draw.ellipse(r, c, 3 * radius, radius, shape=model_image.shape)
            model_image[rr, cc] = axon.id
            
        self.model_image = model_image