#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules with util functions and classes for displaying images with PyQt5.
Created on Wed Jun 19 17:24:26 2019

@author: nicolas
"""

import numpy as np
from skimage import io

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea
from PyQt5.QtGui import QImage, QPixmap

from axoid.utils.image import to_npint

    
def array2pixmap(array):
    """Convert a numpy image array to a QPixmap."""
    img = to_npint(array)
    if img.ndim == 2:  # greyscale
        height, width = img.shape
        bytesPerLine = width
        qIm = QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
    elif img.ndim == 3:
        height, width, channel = img.shape
        bytesPerLine = channel * width
        if channel == 3:  # RGB
            qIm = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        elif channel == 4:  # RGBA
            qIm = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qIm)


class LabelImage(QLabel):
    """
    QLabel for images which rescale with window at defined mode.
    
    Automatically fit current available space with the scaling mode. Adding a
    box frame around seems necessary to get the correct behaviour in some layouts.
    """
    
    def __init__(self, image, *args, framestyle=QLabel.Box, scaling_mode=Qt.KeepAspectRatio, **kwargs):
        """
        Initialize the label with the image.
        
        Parameters
        ----------
        image : str or ndarray
            Path to the image, or the image as a numpy array.
        args : list of arguments
            List of arguments that will be passed to the QLabel constructor.
        framestyle : QLabel.FrameStyle (default = QLabel.Box)
            Frame style for the current widget. Any kind of frame might be 
            necessary to get the correct rescaling behaviour in some layouts.
        scaling_mode : Qt.ScalingMode (default = Qt.KeepAspectRation)
            Rescaling mode for the image, see Qt documentation.
        kwargs : dict of named arguments
            Dictionary of named arguments that will be passed to the QLabel constructor.
        """
        super().__init__(*args, **kwargs)
        self.scaling_mode = scaling_mode
        
        # Initialize the pixmap
        if isinstance(image, str):
            self.pixmap_ = QPixmap(image)
        elif isinstance(image, np.ndarray):
            self.pixmap_ = array2pixmap(image)
        scaledPix = self.pixmap_.scaled(self.size(), self.scaling_mode)
        self.setPixmap(scaledPix)
        
#        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        # Add a frame (required to have correct behaviour in Layouts)
        self.setFrameStyle(framestyle)
    
    def resizeEvent(self, event):
        """Resize the QPixmap with the QLabel."""
        scaledPix = self.pixmap_.scaled(event.size(), self.scaling_mode)
        self.setPixmap(scaledPix)
    
    def update_(self):
        """Rescale the QPixmap and call QLabel update()."""
        scaledPix = self.pixmap_.scaled(self.size(), self.scaling_mode)
        self.setPixmap(scaledPix)
        self.update()
    
    def changeFrame(self, num):
        """To be implemented by child classes."""
        pass


class LabelStack(LabelImage):
    """
    QLabel for stacks of images which rescale with window at defined mode.
    
    Very similar to LabelImage, but allows to use stack of images, and actively
    change the frame to display from inside the stack.
    """
    
    def __init__(self, stack, *args, **kwargs):
        """
        Initialize the label with first image, and keep stack in memory.
        
        Parameters
        ----------
        stack : str or ndarray
            Path to the image stack, or the image stack as a numpy array.
        args : list of arguments
            List of arguments that will be passed to the LabelImage constructor.
        kwargs : dict of named arguments
            Dictionary of named arguments that will be passed to the LabelImage 
            constructor.
        """
        # Initialize the current pixmap
        if isinstance(stack, str):
            self.stack = io.imread(stack)
        elif isinstance(stack, np.ndarray):
            self.stack = stack
        
        super().__init__(self.stack[0], *args, **kwargs)
    
    def changeFrame(self, num):
        """
        Change the stack frame to display.
        
        Parameters
        ----------
        num : int
            Index of the frame in the stack to display.
        """
        size = self.pixmap().size()
        self.pixmap_ = array2pixmap(self.stack[num])
        scaledPix = self.pixmap_.scaled(size, Qt.KeepAspectRatio)
        self.setPixmap(scaledPix)


class VerticalScrollImage(QScrollArea):
    """
    Vertical scroll area with an image that auto-fit the width.
    
    It assumes an image with height > width, or the display will be wrong.
    """
    
    def __init__(self, image, *args, **kwargs):
        """
        Initialize the scroll area with the given image.
        
        Parameters
        ----------
        image : str or ndarray
            Path to the image, or the image as a numpy array.
        args : list of arguments
            List of arguments that will be passed to the QScrollArea constructor.
        kwargs : dict of named arguments
            Dictionary of named arguments that will be passed to the QScrollArea 
            constructor.
        """
        # Initialize ScrollArea
        super().__init__(*args, **kwargs)
        self.setWidgetResizable(True)
        
        # Initialize image widget
        if isinstance(image, str):
            self.pixmap_ = QPixmap(image)
        elif isinstance(image, np.ndarray):
            self.pixmap_ = array2pixmap(image)
        lbl = QLabel()
        lbl.setPixmap(self.pixmap_)
        lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        
        self.setWidget(lbl)
    
    def resizeEvent(self, event):
        """Resize the image to fit the scroll area with fixed ratio."""
        scaledPix = self.pixmap_.scaled(event.size(), Qt.KeepAspectRatioByExpanding)
        self.widget().setPixmap(scaledPix)