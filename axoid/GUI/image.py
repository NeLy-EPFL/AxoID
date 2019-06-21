#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules with util functions and classes for displaying images with PyQt5.
Created on Wed Jun 19 17:24:26 2019

@author: nicolas
"""

import numpy as np
from skimage import io

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QAbstractScrollArea
from PyQt5.QtGui import QImage, QPixmap

from axoid.utils.image import to_npint

    
def _array2pixmap(array):
    """Convert a numpy image array to a QPixmap."""
    img = to_npint(array)
    if img.ndim == 2:
        height, width = img.shape
        bytesPerLine = width
        qIm = QImage(img.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
    elif img.ndim == 3:
        height, width, channel = img.shape
        bytesPerLine = channel * width
        if channel == 3:
            qIm = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        elif channel == 4:
            qIm = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qIm)


class LabelImage(QLabel):
    """QLabel for images which rescale with window at fixed height/width ratio."""
    
    def __init__(self, image, *args, framebox=True, scaling_mode=Qt.KeepAspectRatio, **kwargs):
        """Initialize the label with the image."""
        super().__init__(*args, **kwargs)
        self.scaling_mode = scaling_mode
        
        # Initialize the pixmap
        if isinstance(image, str):
            self._pixmap = QPixmap(image)
        elif isinstance(image, np.ndarray):
            self._pixmap = _array2pixmap(image)
        scaledPix = self._pixmap.scaled(self.size(), self.scaling_mode)
        self.setPixmap(scaledPix)
        
#        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        # Add a frame (required to have correct behaviour in Layouts)
        if framebox:
            self.setFrameStyle(QLabel.Box)
    
    def resizeEvent(self, event):
        """Resize pixmap with the label."""
        scaledPix = self._pixmap.scaled(event.size(), self.scaling_mode)
        self.setPixmap(scaledPix)
    
    def changeFrame(self, num):
        """To be implemented by child classes."""
        pass


class LabelStack(LabelImage):
    """QLabel for stacks of images which rescale with window at fixed height/width ratio."""
    
    def __init__(self, stack, *args, **kwargs):
        """Initialize the label with first image, and keep stack in memory."""
        # Initialize the current pixmap
        if isinstance(stack, str):
            self.stack = io.imread(stack)
        elif isinstance(stack, np.ndarray):
            self.stack = stack
        
        super().__init__(stack[0], *args, **kwargs)
    
    def changeFrame(self, num):
        """Change the stack frame to display."""
        size = self.pixmap().size()
        self._pixmap = _array2pixmap(self.stack[num])
        scaledPix = self._pixmap.scaled(size, Qt.KeepAspectRatio)
        self.setPixmap(scaledPix)


class VerticalScrollImage(QScrollArea):
    """Vertical scroll area with an image that auto-fit the width."""
    
    def __init__(self, image, *args, **kwargs):
        """Initialize the scroll area with the given image."""
        # Initialize ScrollArea
        super().__init__(*args, **kwargs)
        self.setWidgetResizable(True)
        
        # Initialize image widget
        if isinstance(image, str):
            self._pixmap = QPixmap(image)
        elif isinstance(image, np.ndarray):
            self._pixmap = _array2pixmap(image)
        lbl = QLabel()
        lbl.setPixmap(self._pixmap)
        lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        
        self.setWidget(lbl)
    
    def resizeEvent(self, event):
        """Resize the image to fit the scroll area with fixed ratio."""
        scaledPix = self._pixmap.scaled(event.size(), Qt.KeepAspectRatioByExpanding)
        self.widget().setPixmap(scaledPix)