#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains code useful for using internal model for tracking axon identities.
Note that the code is not necessarily robust if not used as intended.
Created on Thu Mar 14 15:03:31 2019

@author: nicolas
"""
### TODOs:
#   1. gcamp signal of axons with 0 or nan when absent ?

import warnings
import numpy as np
from skimage import measure
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# General update_mean function
def _update_mean(update_iter, mean, new_value):
    """Update the online mean with the new value."""
    if mean is None or update_iter == 0:
        return new_value
    new_mean = (mean * update_iter + new_value) / (update_iter + 1)
    return new_mean


class _Axon():
    ### TODOs:
    #   1.
    """Axon object with an identity and properties linked to 2-photon imaging."""
    
    def __init__(self, identity):
        """Create and initialize an axon."""
        # Identity label of the axon
        self.id = identity
        # Row col position to center of mass
        self.position = None
        # Area in pixel
        self.area = None
        # Average tdTomato intensity value
        self.tdTom = None
        # Time serie of GCaMP intensity values
        self.gcamp = np.array([])
        # Average of shapes
        self.shape = np.array([[]])
        # Iteration of update (useful for moving averages)
        self._update_iter = 0
    
    def update_axon(self, id_frame, id_seg, region):
        """Update the axon's properties with the given identity frame."""
        # Print a warning if identity is different than region's label
        if self.id != region.label:
            warnings.warn("Axon %d is updated with a region labelled %d." % (self.id, region.label), 
                          RuntimeWarning)
        
        roi = id_seg == region.label
        local_CoM = np.mean(np.nonzero(id_seg), 1)
        
        position = np.array(region.centroid) - local_CoM
        self.position = _update_mean(self._update_iter, self.position, position)
        self.area = _update_mean(self._update_iter, self.area, region.area)
        self.tdTom = _update_mean(self._update_iter, self.tdTom, id_frame[roi, 0].mean())
        self.gcamp = np.append(self.gcamp, id_frame[roi, 1].mean())
        
        # Find shape array around centroid and update it
        local_h, local_w = region.image.shape
        r_s = max(region.local_centroid[0], local_h - region.local_centroid[0])
        c_s = max(region.local_centroid[1], local_w - region.local_centroid[1])
        new_shape = np.pad(region.image, [(int(r_s - region.local_centroid[0] + 0.5), 
                                           int(r_s - (local_h - region.local_centroid[0]) + 0.5)),
                                          (int(c_s - region.local_centroid[1] + 0.5), 
                                           int(c_s - (local_w - region.local_centroid[1]) + 0.5))],
                           'constant')
        self._update_shape(new_shape)
        self._update_iter += 1
    
    def _update_shape(self, new_shape):
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
        self.shape = _update_mean(self._update_iter, shapes[0], shapes[1])
    
    def __repr__(self):
        """Return the representation of the axon object (cannot be evaluated)."""
        return "%s(id=%r, position=%r, area=%r, tdTom=%r, len(gcamp)=%r, shape.shape=%r)" % \
            (self.__class__.__name__, self.id, self.position, self.area, self.tdTom,
             len(self.gcamp), self.shape.shape)
    
    def __str__(self):
        """Return a formatted string of the axon object representation."""
        return "Axon %d: position=%s, area=%d, tdTom=%.3f, len(gcamp)=%d, shape.shape=%s" % \
            (self.id, self.position, self.area, self.tdTom, len(self.gcamp), self.shape.shape)


class InternalModel():
    ### TODOs:
    #   1. Take into account shift of local CoM with more/less neurons
    """Model of the axon structure of 2-photon images."""
    
    def __init__(self, add_new_axons=True):
        """Create and initialize an internal model."""
        # Container of axons object
        self.axons = []
        # Boolean deciding if new axons are labelled and added to the model or discarded
        self.add_new_axons = add_new_axons
        # Image of the model's axons
        self.image = None
        # Coordinate of the center of mass
        self.center_of_mass = None
        # Iteration of update (useful for moving averages)
        self._update_iter = 0
    
    def initialize(self, id_frame, id_seg, add_new_axons=None):
        """(Re-)Initialize the model with the given identity frame."""
        # Re-initialize the mdoel
        self.axons = []
        if add_new_axons is not None:
            self.add_new_axons = add_new_axons
        self.image = np.zeros(id_seg.shape, np.uint8)
        self._update_iter = 1
        
        # If id_seg is only segmentation (boolean), create labels
        if id_seg.dtype == np.bool:
            id_seg = measure.label(id_seg, connectivity=1)
        
        regions = measure.regionprops(id_seg, coordinates='rc')
        self.center_of_mass = np.mean(np.nonzero(id_seg), 1)
        
        # Add an axon per ROI
        for region in regions:
            axon = _Axon(region.label)
            axon.update_axon(id_frame, id_seg, region)
            self.axons.append(axon)
            
        # Draw the model
        self._draw_model()
    
    def match_frame(self, frame, seg):
        """Match axons of the given frame to the model's (new axons are assigned new labels)."""
        # Extract ROIs and compute local center of mass
        labels = measure.label(seg, connectivity=1)
        regions = measure.regionprops(labels, coordinates='rc')
        centroids = np.array([region.centroid for region in regions])
        local_CoM = np.mean(np.nonzero(seg), 1)
        
        # Build a cost matrix where rows are new ROIs and cols are model's axons
        # Currently, cost = distance
        x_roi = centroids - local_CoM
        x_model = np.stack([axon.position for axon in self.axons], axis=0)
        cost_matrix = cdist(x_roi, x_model)
        
        # Assign identities through the hungarian method
        rows_ids, col_ids = linear_sum_assignment(cost_matrix)
        ids = dict(zip([regions[i].label for i in rows_ids],
                       [self.axons[i].id for i in col_ids]))
        # Unassigned (new) axons are given new labels or are discarded as background
        for i in range(len(regions)):
            if i in rows_ids:
                continue
            if self.add_new_axons:
                # Create new axon id
                new_id = 1
                while new_id in [axon.id for axon in self.axons]:
                    new_id += 1
                ids.update({regions[i].label: new_id})
            else:
                ids.update({regions[i].label: 0})
        
        # Make identity image, and return it
        identities = np.zeros(seg.shape, np.uint8)
        for region in regions:
            roi = labels == region.label
            identities[roi] = ids[region.label]
        return identities
    
    def update_model(self, id_frame, id_seg):
        """Update the model based on the identity frame (new labels are added as new axons)."""
        # Axons' identity to index mapping
        id_to_idx = dict()
        for i in range(len(self.axons)):
            id_to_idx.update({self.axons[i].id: i})
        
        regions = measure.regionprops(id_seg, coordinates='rc')
        
        for region in regions:
            # If new axon, add it to the model or discard it
            if region.label not in [axon.id for axon in self.axons]:
                if self.add_new_axons:
                    axon = _Axon(region.label)
                    axon.update_axon(id_frame, id_seg, region)
                    self.axons.append(axon)
            # Else, update it
            else:
                idx = id_to_idx[region.label]
                self.axons[idx].update_axon(id_frame, id_seg, region)
        
        local_CoM = np.mean(np.nonzero(id_seg), 1)
        self.center_of_mass = _update_mean(self._update_iter, self.center_of_mass, local_CoM)
        self._update_iter += 1
        # Update the model image
        self._draw_model()
    
    def _draw_model(self):
        """Update the model image based on the axons' properties."""
        model_image = np.zeros_like(self.image)
        
        for axon in self.axons:
            # Create binary axon segmentation from axon shape and area
            axon_seg = axon.shape >= np.sort(axon.shape, axis=None)[- int(axon.area + 0.5)]
            # Make sure that it is odd-sized
            axon_seg = np.pad(axon_seg, [(0, 1 - axon_seg.shape[0] % 2), 
                                         (0, 1 - axon_seg.shape[1] % 2)], 'constant')
            
            # Center of the axon on the image
            center = (axon.position + self.center_of_mass).astype(np.uint16)
            # Compute row and col coordinates to fit the axon in the image
            a_min_row = max(0, axon_seg.shape[0] // 2 - center[0])
            a_min_col = max(0, axon_seg.shape[1] // 2 - center[1])
            a_max_row = min(axon_seg.shape[0], axon_seg.shape[0] // 2 + model_image.shape[0] - center[0])
            a_max_col = min(axon_seg.shape[1], axon_seg.shape[1] // 2 + model_image.shape[1] - center[1])
            
            # Compute the axon position in the image
            min_row = max(0, center[0] - axon_seg.shape[0] // 2)
            min_col = max(0, center[1] - axon_seg.shape[1] // 2)
            max_row = min(model_image.shape[0], center[0] + axon_seg.shape[0] // 2 + 1)
            max_col = min(model_image.shape[1], center[1] + axon_seg.shape[1] // 2 + 1)
            model_image[min_row:max_row, min_col:max_col] += \
                (axon_seg[a_min_row:a_max_row, a_min_col:a_max_col] * axon.id).astype(model_image.dtype)
                
        self.image = model_image
    
    def __repr__(self):
        """Return the representation of the internal model (cannot be evaluated)."""
        return "%s(add_new_axons=%r, axons=%r)" % \
            (self.__class__.__name__, self.add_new_axons, self.axons)
    
    def __str__(self):
        """Return a formatted string of the internal model representation."""
        string = "Internal model with %d axons: (new axons are %s)" % \
            (len(self.axons), "added" if self.add_new_axons else "discarded")
        
        for axon in self.axons:
            string += "\n  - " + str(axon)
        
        return string