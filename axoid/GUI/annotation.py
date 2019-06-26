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

from .multipage import AxoidPage


class AnnotationPage(AxoidPage):
    """
    Page of the manual annotation process.
    
    Here, the user creates manual annotations for some selected frames 
    to then fine tune the network on. The process can be repeated as will.
    Finally, the rest of the AxoID pipeline will be applied to the data.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the annotation page.
        
        Args and kwargs are passed to the AxoidPage constructor.
        """
        super().__init__(*args, **kwargs)
    
    def initUI(self):
        """Initialize the interface."""