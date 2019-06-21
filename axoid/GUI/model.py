#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the model correction window widget.
This GUI page allows the user to modify the model of the tracker to improve the 
axon's identities (e.g.: by fusing, cutting, or discarding ROIs).
Created on Wed Jun 19 11:48:44 2019

@author: nicolas
"""

from .multipage import PageWidget


class ModelPage(PageWidget):
    """Page of the model correction process."""
    
    def __init__(self, experiment, *args, **kwargs):
        """Initialize the model page."""
        super().__init__(*args, **kwargs)
        
        self.experiment = experiment