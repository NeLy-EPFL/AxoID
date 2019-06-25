#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module with useful functions for the tracking.
Created on Mon May  6 11:49:30 2019

@author: nicolas
"""

import numpy as np


def get_rules(image, remove_contradictions=True):
    """
    Return the row-order rules between the labelled regions as a 2D array.
    
    Parameters
    ----------
    image : ndarray
        Identity image, where ROI have an ID corresponding to their pixels value.
    remove_contradictions : bool (default = True)
        If True and two rules contradict them-selves (e.g. 1 left to 2 and 
        2 left to 1), both are removed.
    
    Returns
    -------
    rules : ndarray
        Array of rules, where each row is a rule. A rule is two numbers, which
        indicates that the first one should appear left to the second one.
        Can be empty if there are no rows where there are multiple labels.
    """
    rules = []
    
    # Find pixels where the label changes horizontally
    mask = np.zeros(image.shape, dtype=np.bool)
    mask[:, 0] = image[:, 0] != 0
    mask[:, 1:] = np.logical_and(image[:, 1:] != image[:, :-1], image[:, 1:] != 0)
    
    # Create rules by taking pairs of row-ordered labels
    for i in range(image.shape[0]):
        order = image[i, mask[i]]
        if len(order) <= 1:
            continue
        for j in range(0, len(order) - 1):
            for k in range(j + 1, len(order)):
                # Do not add "self-rules" (e.g.: (1,1))
                if order[j] != order[k]:
                    rules.append((order[j], order[k]))
    
    # Stop there if no rules were found
    if rules == []:
        return np.array([])
    # Or remove duplicates
    else:
        rules = np.unique(rules, axis=0)
    
    # Optionally remove contradicting rules (e.g.: (1,2) & (2,1) --> N/A)
    if remove_contradictions:
        to_remove = []
        for i in range(len(rules)):
            for j in range(i + 1, len(rules)):
                if (rules[i] == rules[:, ::-1][j]).all():
                    to_remove.append(i)
                    to_remove.append(j)
        rules = rules[[i for i in range(len(rules)) if i not in to_remove]]
    
    return rules


def _verify_frame(rules, label):
    """Verify if the frame violates any rules."""
    # If no rules, cannot be wrong
    if rules.size == 0:
        return False
    orders = get_rules(label)
    verification = np.array([(rules == order[::-1]).all(1) for order in orders])
    return np.any(verification)

def rules_violated(rules, labels):
    """
    Verify if the labels violates any rules.
    
    Parameters
    ----------
    rules : ndarray
        Array of rules as returned by get_rules().
    labels : ndarray
        Label image or stack of label images.
    
    Returns
    -------
    verification : bool or array of bool
        False if no rules are violated, True if any is for the image.
        If labels is a stack, the array is the verification for each single image.
    """    
    if labels.ndim == 2:
        return _verify_frame(rules, labels)
    else:   
        return np.array([_verify_frame(rules, label) for label in labels])
    

def renumber_ids(model, identities=None):
    """
    Renumber axons with consecutive integers starting at 1.
    
    Parameters
    ----------
    model : InternalModel
        An internal model used for tracking. Its axon identities will be 
        renumbered accordingly, and its image redrawn.
    identities : ndarray (optional)
        Stack of label images corresponding to tracked identities with the model.
    
    Returns
    -------
    [new_identities] : ndarray (optional)
        If identities were given, new_identities will be returned and contained
        the renumbered identities.
        
    """
    # Change numbering in model and keep track of it for identities
    new_ids = dict()
    for i, axon in enumerate(model.axons):
        new_ids.update({axon.id: i + 1})
        axon.id = i + 1
    # Redraw the model
    model._draw()
    
    # Change identities if applicable
    if identities is not None:
        new_identities = np.zeros_like(identities)
        for old_id, new_id in new_ids.items():
            new_identities[identities == old_id] = new_id
        return new_identities