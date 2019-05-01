#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains code useful for using internal model for tracking axon identities.
Note that the code is not necessarily robust if not used as intended.
Created on Thu Mar 14 15:03:31 2019

@author: nicolas
"""

import warnings
import numpy as np
#TODO: remove after testing
import matplotlib
import matplotlib.pyplot as plt
from skimage import measure
from skimage import morphology as morph
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# General update_mean function
def _update_mean(update_iter, mean, new_value):
    """Update the online mean with the new value."""
    if mean is None or update_iter == 0:
        return new_value
    new_mean = (mean * update_iter + new_value) / (update_iter + 1)
    return new_mean


class _GCaMP(list):
    """List for GCaMP intensities which returns np.nan when out of bounds."""
    
    def __getitem__(self, idx):
        """If index is out of bound, return np.nan."""
        if idx < len(self):
            return super().__getitem__(idx)
        else:
            return np.nan
    
    def __setitem__(self, idx, value):
        """If index is out of bound, fills the list with np.nan to fit the index, then set the new item."""
        if idx < len(self):
           super().__setitem__(idx, value)
        else: # new index, need to increase gcamp list
            self.extend([np.nan] * (idx + 1 - len(self)))
            super().__setitem__(idx, value)


class _Axon():
    """Axon object with an identity and properties linked to 2-photon imaging."""
    
    def __init__(self, identity):
        """Create and initialize an axon."""
        # Identity label of the axon
        self.id = identity
        # Row col position to center of mass
        self.position = None
        # Area in pixel (mainly for drawing purposes)
        self.area = None
        # Area normalized by the number of ROI pixel of the frame
        self.area_norm = None
        # Average tdTomato intensity value
        self.tdTom = None
        # Time serie of GCaMP intensity values
        self.gcamp = _GCaMP()
        # Average of shapes
        self.shape = np.array([[]])
        # Iteration of update (useful for moving averages)
        self._update_iter = 0
    
    def update_axon(self, id_frame, id_seg, region, time_idx=None):
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
        self.area_norm = _update_mean(self._update_iter, self.area_norm, region.area / (id_seg != 0).sum())
        self.tdTom = _update_mean(self._update_iter, self.tdTom, id_frame[roi, 0].mean())
        if time_idx is not None:
            self.gcamp[time_idx] = id_frame[roi, 1].mean()
        
        # Find shape array around centroid and update it
        local_h, local_w = region.image.shape
        r_s = max(region.local_centroid[0], local_h - region.local_centroid[0])
        c_s = max(region.local_centroid[1], local_w - region.local_centroid[1])
        new_shape = np.pad(region.image, 
                           [(int(r_s - region.local_centroid[0] + 0.5), 
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
        return "%s(id=%r, position=%r, area=%r, area_norm=%r, tdTom=%r, len(gcamp)=%r, shape.shape=%r)" % \
            (self.__class__.__name__, self.id, self.position, self.area, self.area_norm,
             self.tdTom, len(self.gcamp), self.shape.shape)
    
    def __str__(self):
        """Return a formatted string of the axon object representation."""
        return "Axon %d: position=%s, area=%d, area_norm=%.3f, tdTom=%.3f, len(gcamp)=%d, shape.shape=%s" % \
            (self.id, self.position, self.area, self.area_norm, self.tdTom, 
             len(self.gcamp), self.shape.shape)


class InternalModel():
    """Model of the axon structure of 2-photon images."""
    
    def __init__(self):
        """Create and initialize an internal model."""
        # Container of axons object
        self.axons = []
        # Normalization factors for axons properties
        self.norm = None
        # Image of the model's axons
        self.image = None
        # Coordinate of the center of mass
        self.center_of_mass = None
        # Iteration of update (useful for moving averages)
        self._update_iter = 0
        # Weights for the data normalization and thresholding in Hungarian assignment
        self.W_POSITION = 1.0
        self.W_AREA = 0.5
        self.W_TDTOM = 1.0
        self.W_GCAMP = 1.0
        # Threshold for Hungarian assignment (weighted by previous weights)
        self.TH_HUNGARIAN = 1.0
    
    def fit_normalization(self, rgb_stack, seg_stack):
        """Compute center and deviation of ROI properties for normalization purposes."""
        self.norm = dict()
        positions = []
        areas = []
        tdToms = []
        gcamps = []
        
        # Loop on frames
        prev_labels = None
        prev_regions = None
        for i in range(len(rgb_stack)):
            labels = measure.label(seg_stack[i], connectivity=1)
            regions = measure.regionprops(labels, coordinates='rc')
            local_CoM = np.mean(np.nonzero(labels), 1)
            
            # Loop on ROI
            for region in regions:
                # Row col coordinates w.r.t. center of mass
                positions.append(np.array(region.centroid) - local_CoM)
                # Area normalized by number of ROI pixel in the frame
                areas.append(region.area / seg_stack[i].sum())
                # tdTom intensity
                tdToms.append(rgb_stack[i, labels == region.label, 0].mean())
                # GCaMP decrease (cannot be negative)
                if i > 0:
                    for prev_region in prev_regions:
                        gcamp_decrease = - (rgb_stack[i, labels == region.label, 0].mean() - \
                                            rgb_stack[i-1, prev_labels == prev_region.label, 0].mean())
                        if gcamp_decrease >= 0:
                            gcamps.append(gcamp_decrease)
            prev_labels = labels.copy()
            prev_regions = regions.copy()
        
        # Sets mean and standard deviation (with a minimum to reduce influence
        # of stochasticity on cost)
        self.norm["position"] = dict()
        self.norm["position"]["mean"] = np.mean(positions, axis=0)
        self.norm["position"]["std"] = np.maximum(10.0, np.std(positions, axis=0))
        
        self.norm["area"] = dict()
        self.norm["area"]["mean"] = np.mean(areas)
        self.norm["area"]["std"] = max(0.15, np.std(areas))
        
        self.norm["tdTom"] = dict()
        self.norm["tdTom"]["mean"] = np.mean(tdToms)
        self.norm["tdTom"]["std"] = max(0.1, np.std(tdToms))
        
        self.norm["gcamp"] = dict()
        # Already positive, no centering as it will directly serves as distance
        self.norm["gcamp"]["mean"] = None
        self.norm["gcamp"]["std"] = max(0.1, np.std(gcamps))
    
    def initialize(self, id_frame, id_seg, time_idx=None):
        """(Re-)Initialize the model with the given identity frame."""
        # Re-initialize the model, except for normalization
        if self.norm is None:
            raise RuntimeError("The model needs to be fitted a normalization before its initialization.")
        self.axons = []
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
            axon.update_axon(id_frame, id_seg, region, time_idx=time_idx)
            self.axons.append(axon)
            
        # Draw the model
        self._draw_model()
    
    def match_frame(self, frame, seg, time_idx=None, max_iter=5, debug=False, return_debug=False):
        """Match axons of the given frame to the model's."""
        # Hyper-parameters
        NORM_DIST = min(seg.shape) # normalization of the distances
        DIST_45 = 0.1 * NORM_DIST # distance at which the normalization angle equals 45°
        W_DIST = 1.0
        W_ANGLE = 1.0
        W_AREA = 1.0
        TH_DUMMY = 0.15
        # If no ROI, do nothing
        if seg.sum() == 0:
            return np.zeros(seg.shape, np.uint8)
        # One iteration of assignment minimum
        max_iter = max(1, max_iter)
        if debug:
            id_cmap = matplotlib.cm.get_cmap('viridis')
            id_cmap.set_under([0,0,0])
        
        # Extract ROIs and compute local center of mass
        labels = measure.label(seg, connectivity=1)
        regions = measure.regionprops(labels, coordinates='rc')
        centroids = np.array([region.centroid for region in regions])
        areas = np.array([region.area for region in regions])
        
        # Construct the outer cost matrix by looping over ROIs and axons
        cost_matrix = np.zeros((len(regions), len(self.axons)))
        for i in range(len(regions)):
            for k in range(len(self.axons)):
                # Construct the inner cost matrix (without taking ROI i and axon k in account)
                # Distance cost
                x_roi = np.delete(centroids, i , 0) - centroids[i]
                x_model = np.stack([self.axons[j].position for j in range(len(self.axons)) if j != k],
                                    axis=0) - self.axons[k].position
                cost_dist = cdist(x_roi, x_model) / NORM_DIST
                
                # Hungarian assignment method
                in_cost_matrix = W_DIST * cost_dist
                in_cost_matrix = np.concatenate([in_cost_matrix, 
                      np.ones((len(regions) - 1,) * 2) * max(TH_DUMMY, in_cost_matrix.min())], axis=1)
                row_ids, col_ids = linear_sum_assignment(in_cost_matrix)
                
                # Cost of outer cost matrix is equal to average of inner cost
                cost_matrix[i, k] = np.mean(in_cost_matrix[row_ids, col_ids])
        
        # Add "dummy" axons in the cost matrix for axons not in the model
        cost_matrix = np.concatenate([cost_matrix, 
              np.ones((len(regions), len(regions))) * max(TH_DUMMY, cost_matrix.min())], axis=1)
        
        # Assign identities through the hungarian method
        row_ids, col_ids = linear_sum_assignment(cost_matrix)
        
        # Build a dictionary mapping ROI labels to axon IDs
        ids = dict()
        for i, region in enumerate(regions):
            c_idx = col_ids[np.where(row_ids == i)[0]]
            if i in row_ids and c_idx < len(self.axons):
                ids.update({region.label: self.axons[int(c_idx)].id})
            else:
                ids.update({region.label: 0})
        
        if debug:
            print(cost_matrix[:,:len(self.axons)], cost_matrix[0,len(self.axons)], TH_DUMMY)
            print(row_ids, col_ids, ids)
            print(cost_matrix[row_ids, col_ids],
                  np.mean([cost_matrix[row_ids[j], col_ids[j]]
                  for j in range(len(row_ids)) if col_ids[j] < len(self.axons)]))
            identities = np.zeros(seg.shape, np.uint8)
            for region in regions:
                roi = labels == region.label
                identities[roi] = ids[region.label]
            plt.figure(figsize=(8,4))
            plt.subplot(121); plt.title("model")
            plt.imshow(self.image, cmap=id_cmap, vmin=1, vmax=self.image.max())
            plt.subplot(122); plt.title("identity")
            plt.imshow(identities, cmap=id_cmap, vmin=1, vmax=self.image.max())
            plt.show()
        if return_debug:
            return cost_matrix, row_ids, col_ids
        # Make identity image, and return it
        identities = np.zeros(seg.shape, np.uint8)
        for region in regions:
            roi = labels == region.label
            identities[roi] = ids[region.label]
        return identities
    
    def update_model(self, id_frame, id_seg, time_idx=None):
        """Update the model based on the identity frame (new labels are added as new axons)."""
        # Axons' identity to index mapping
        id_to_idx = {self.axons[i].id: i for i in range(len(self.axons))}
        
        regions = measure.regionprops(id_seg, coordinates='rc')
        
        for region in regions:
            # If axon in model, update it
            if region.label in [axon.id for axon in self.axons]:
                idx = id_to_idx[region.label]
                self.axons[idx].update_axon(id_frame, id_seg, region, time_idx=time_idx)
        
        # If time index is given, unset the GCaMP signal of absent neurons
        if time_idx is not None:
            frame_ids = [region.label for region in regions]
            for axon in self.axons:
                if axon.id not in frame_ids:
                    axon.gcamp[time_idx]= np.nan
        
        # If no ROI, don't update center of mass nor redraw the model
        if id_seg.sum() == 0:
            return
        
        local_CoM = np.mean(np.nonzero(id_seg), 1)
        self.center_of_mass = _update_mean(self._update_iter, self.center_of_mass, local_CoM)
        self._update_iter += 1
        # Update the model image
        self._draw_model()
    
    def _draw_model(self):
        """Update the model image based on the axons' properties."""
        model_image = np.zeros((len(self.axons), ) + self.image.shape, self.image.dtype)
        
        # First create an image by axon
        for i, axon in enumerate(self.axons):
            # Create binary axon segmentation from axon shape and area
            axon_seg = axon.shape >= np.sort(axon.shape, axis=None)[- int(axon.area + 0.5)]
            # Make sure that it is odd-sized
            axon_seg = np.pad(axon_seg, [(0, 1 - axon_seg.shape[0] % 2), 
                                         (0, 1 - axon_seg.shape[1] % 2)], 'constant')
            
            # Center of the axon on the image
            center = (axon.position + self.center_of_mass).astype(np.uint16)
            # Compute row and col coordinates to fit the axon in the image
            a_min_row = (axon_seg.shape[0] // 2 - center[0]).clip(0, axon_seg.shape[0])
            a_min_col = (axon_seg.shape[1] // 2 - center[1]).clip(0, axon_seg.shape[1])
            a_max_row = (axon_seg.shape[0] // 2 + model_image.shape[1] - center[0]).clip(0, axon_seg.shape[0])
            a_max_col = (axon_seg.shape[1] // 2 + model_image.shape[2] - center[1]).clip(0, axon_seg.shape[1])
            
            # Compute the axon position in the image
            min_row = (center[0] - axon_seg.shape[0] // 2).clip(0, model_image.shape[1])
            min_col = (center[1] - axon_seg.shape[1] // 2).clip(0, model_image.shape[2])
            max_row = (center[0] + axon_seg.shape[0] // 2 + 1).clip(0, model_image.shape[1])
            max_col = (center[1] + axon_seg.shape[1] // 2 + 1).clip(0, model_image.shape[2])
            model_image[i, min_row:max_row, min_col:max_col] = \
                (axon_seg[a_min_row:a_max_row, a_min_col:a_max_col] * axon.id).astype(model_image.dtype)
        
#        # Then reduce them to one, while adding borders if some axons overlap
#        # Create distance images for each axon
#        dist = []
#        for i in range(model_image.shape[0]):
#            if np.count_nonzero(model_image[i]) == 0:
#                continue
#            dist.append(ndi.distance_transform_edt(model_image[i]))
#        dist = np.array(dist)
#
#        # Centroid of axons for watershed starting points
#        local_maxi = np.zeros(self.image.shape, dtype=np.uint8)
#        for i in range(dist.shape[0]):
#            r, c = self.axons[i].position + self.center_of_mass
#            local_maxi[int(r), int(c)] = self.axons[i].id
#        
#        # Watershed with borderlines
#        with warnings.catch_warnings():
#            warnings.simplefilter("ignore")
#            model_image = morph.watershed(-dist.max(0), local_maxi, 
#                                          mask=model_image.max(0).astype(np.bool), 
#                                          watershed_line=True)
        
        self.image = model_image.max(0)
    
    def __repr__(self):
        """Return the representation of the internal model (cannot be evaluated)."""
        return "%s(axons=%r)" % \
            (self.__class__.__name__, self.axons)
    
    def __str__(self):
        """Return a formatted string of the internal model representation."""
        string = "Internal model with %d axons:" % len(self.axons)
        
        for axon in self.axons:
            string += "\n  - " + str(axon)
        
        return string