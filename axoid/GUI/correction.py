#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the frame-wise correction window widget.
The user can modify the results on a frame by frame basis on this page, e.g. by 
changing the identities of ROIs, and/or correcting the segmentations.
Created on Wed Jun 19 11:48:44 2019

@author: nicolas
"""

from .multipage import AxoidPage


class CorrectionPage(AxoidPage):
    """
    Page of the frame-wise correction process.
    
    Here, the user can modify the results on a frame by frame basis, e.g. by 
    changing the identities of ROIs, and/or correcting the segmentation.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the correction page.
        
        Args and kwargs are passed to the AxoidPage constructor.
        """
        super().__init__(*args, **kwargs)
    
    def initUI(self):
        """Initialize the interface."""