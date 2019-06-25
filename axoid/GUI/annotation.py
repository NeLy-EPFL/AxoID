#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the manual annotation window widget.
This page lets the user create manually annotations for some selected frames 
to then fine tune the network on. The process can be repeated as will.
Finally, the rest of the AxoID pipeline will be applied to the data.
Created on Wed Jun 19 11:48:45 2019

@author: nicolas
"""

from .multipage import PageWidget


class AnnotationPage(PageWidget):
    """Page of the manual annotation process."""
    
    def __init__(self, experiment, *args, **kwargs):
        """Initialize the annotation page."""
        super().__init__(*args, **kwargs)
        
        self.experiment = experiment