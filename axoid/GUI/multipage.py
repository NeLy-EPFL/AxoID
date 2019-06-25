#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base PyQt classes for multi-page widgets with change through signal emission.
Page's identifiers are considered to be integers.
Created on Wed Jun 19 11:13:27 2019

@author: nicolas
"""

import os

from skimage import io

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (QWidget, QStackedWidget, QHBoxLayout, QVBoxLayout,
                             QGridLayout, QLabel, QLineEdit, QSlider)


class PageWidget(QWidget):
    """PyQt widget corresponding to a single page for multi-page widgets."""
    changedPage = pyqtSignal(int)
    quitApp = pyqtSignal()
    
    def initUI(self):
        """Function to be implemented in child classes to create the UI."""
    
    def emitChangedPage(self, page):
        """Return a function to emit a changedPage signal to the given page."""
        def emit(state):
            self.changedPage.emit(page)
        return emit

class MultiPageWidget(QWidget):
    """Base PyQt widget for multi-pages."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the widget."""
        super().__init__(*args, **kwargs)
        
        # Initialize widget stack
        self.stackedWidget = QStackedWidget(self)
        self.pageWidgets = {}
    
    def addPage(self, pageWidget, pageID, *args, **kwargs):
        """Add the page."""
        pageWidget.changedPage.connect(self.changePage)
        pageWidget.quitApp.connect(self.close)
        self.stackedWidget.addWidget(pageWidget, *args, **kwargs)
        self.pageWidgets.update({pageID: pageWidget})
    
    def removePage(self, pageID, *args, **kwargs):
        """Remove the page."""
        self.stackedWidget.removeWidget(self.pageWidgets[pageID])
        self.pageWidgets.pop(pageID)
    
    def changePage(self, pageID):
        """Change the current page."""
        self.stackedWidget.setCurrentWidget(self.pageWidgets[pageID])
        self.pageWidgets[pageID].initUI()


class AxoidPage(PageWidget):
    """Page with base layouts for AxoID GUI."""
    
    def __init__(self, experiment, *args, **kwargs):
        """Initialize the selection page."""
        super().__init__(*args, **kwargs)
        self.experiment = experiment
        
        self.len_exp = len(io.imread(os.path.join(self.experiment, "2Pimg", "RGB.tif")))
        # Layout/widget for displaying a choice of image
        self.choice_layouts = dict()
        self.choice_widgets = dict()
        
        # Initialize UI
        # Top level layout
        self.top_hbox = QHBoxLayout()
        
        ## Left layout with images from all outputs
        self.left_vbox = QVBoxLayout()
        self.left_display = QGridLayout()
        self.left_display.setAlignment(Qt.AlignBottom)
        # Bottom slider to change displayed frame
        nframe_hbox = QHBoxLayout()
        nframe_hbox.setAlignment(Qt.AlignTop)
        nframe_hbox.addWidget(QLabel("Frame:"))
        self.nframe_edit = QLineEdit()
        self.nframe_edit.setFocusPolicy(Qt.ClickFocus)
        self.nframe_edit.setText(str(0))
        self.nframe_edit.setMaxLength(4)
        self.nframe_edit.setMaximumWidth(40)
        self.nframe_edit.setValidator(QIntValidator(0, self.len_exp - 1))
        self.nframe_edit.editingFinished.connect(
                lambda: self.changeFrame(self.nframe_edit.text()))
        nframe_hbox.addWidget(self.nframe_edit)
        nframe_hbox.addStretch(1)
        self.nframe_sld = QSlider(Qt.Horizontal)
        self.nframe_sld.setFocusPolicy(Qt.ClickFocus)
        self.nframe_sld.setTickInterval(100)
        self.nframe_sld.setTickPosition(QSlider.TicksBothSides)
        self.nframe_sld.setMinimum(0)
        self.nframe_sld.setMaximum(self.len_exp - 1)
        self.nframe_sld.valueChanged[int].connect(self.changeFrame)
        nframe_hbox.addWidget(self.nframe_sld, stretch=9)
        nframe_hbox.addStretch(1)
        
        self.left_vbox.addLayout(self.left_display, stretch=1)
        self.left_vbox.addLayout(nframe_hbox, stretch=0)
        
        ## Right layout with buttons and actions
        self.right_control = QVBoxLayout()
        
        self.top_hbox.addLayout(self.left_vbox, stretch=1)
        self.top_hbox.addLayout(self.right_control, stretch=0)
        
        self.setLayout(self.top_hbox)
    
    def changeFrame(self, num):
        """Change the displayed frame num."""
        self.nframe_edit.setText(str(num))
        self.nframe_sld.setSliderPosition(int(num))