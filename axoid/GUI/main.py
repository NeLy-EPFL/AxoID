#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module of AxoID user correction GUI with the main window widget.
Created on Tue Jun 18 17:16:11 2019

@author: nicolas
"""

import os.path

from PyQt5.QtWidgets import QHBoxLayout

from .multipage import MultiPageWidget
from .selection import SelectionPage
from .model import ModelPage
from .correction import CorrectionPage
from .annotation import AnnotationPage
from .constants import PAGE_SELECTION, PAGE_MODEL, PAGE_CORRECTION, PAGE_ANNOTATION


class AxoIDWindow(MultiPageWidget):
    """Main window for the AxoID GUI."""
    
    def __init__(self, experiment, *args, **kwargs):
        """Initialize the window."""
        super().__init__(*args, **kwargs)
        
        self.experiment = experiment
        
        # Set pages
        self.addPage(SelectionPage(self.experiment), PAGE_SELECTION)
        self.addPage(ModelPage(self.experiment), PAGE_MODEL)
        self.addPage(CorrectionPage(self.experiment), PAGE_CORRECTION)
        self.addPage(AnnotationPage(self.experiment), PAGE_ANNOTATION)
        
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface."""
        # Set layout
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.stackedWidget)
        self.setLayout(main_layout)
        
        # Set title and default shape
        self.setWindowTitle("AxoID user correction")
        width, height = 1920/2, 1080/2
        self.wh_ratio = width / height  # 16:9
        self.resize(width, height)
        self.show()
        