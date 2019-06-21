#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the output selection window widget.
In this GUI page, the use will have to select which data to continue with between 
raw data, cross-correlation registered data, and optic flow warped data.
Created on Wed Jun 19 11:48:44 2019

@author: nicolas
"""

import os.path

import numpy as np
from skimage import io
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QPixmap
from PyQt5.QtWidgets import (QHBoxLayout, QVBoxLayout, QGridLayout,
                             QPushButton, QLabel, QComboBox, QSlider, QSizePolicy,
                             QGroupBox, QLineEdit, QLayout, QScrollArea, QAbstractScrollArea)

from .constants import CHOICE_PATHS
from .multipage import PageWidget
from .image import LabelImage, LabelStack, VerticalScrollImage
from axoid.utils.image import to_id_cmap


class SelectionPage(PageWidget):
    """Page of the output selection process."""
    
    def __init__(self, experiment, *args, **kwargs):
        """Initialize the selection page."""
        super().__init__(*args, **kwargs)
        
        # Initialize experiment and image stacks
        self.experiment = experiment
        self.n_axons_max = -np.inf
        self.models = dict()
        self.roi_autos = dict()
        self.choice_layouts = dict()
        self.choice_widgets = dict()
        for folder in ["raw", "ccreg", "warped"]:
            model_path = os.path.join(self.experiment, "output", "axoid_internal",
                                      folder, "model.tif")
            model_img = io.imread(model_path)
            if model_img.max() > self.n_axons_max:
                self.n_axons_max = model_img.max()
            model_img = to_id_cmap(model_img)
            model_lbl = LabelImage(model_img)
            self.models.update({folder: model_lbl})
            
            roi_path = os.path.join(self.experiment, "output", "ROI_auto",
                                    folder, "RGB_seg.tif")
            roi_img = io.imread(roi_path)
            roi_lbl = LabelStack(roi_img)
            self.roi_autos.update({folder: roi_lbl})
            
            self.choice_layouts.update({folder: QHBoxLayout()})
            self.choice_widgets.update({folder: QLabel()})
            self.choice_layouts[folder].addWidget(self.choice_widgets[folder],
                                                  alignment=Qt.AlignCenter)
        self.len_exp = len(io.imread(roi_path))
        
        self.initUI()
    
    def initUI(self):
        """Initialize the interface."""
        # Top level layout
        top_hbox = QHBoxLayout()
        
        ## Left layout with images from all outputs
        left_vbox = QVBoxLayout()
        left_display = QGridLayout()
        for row in [1, 2, 3]:
            left_display.setRowStretch(row, 1)
        for col in [1, 2, 3]:
            left_display.setColumnStretch(col, 1)
        left_vbox.addLayout(left_display)
            
        # Row and column labels
        labels = ["Raw", "CCReg", "Warped", "Model", "ROI_auto"]
        positions = [(0,1), (0,2), (0,3), (1,0), (2,0)]
        alignments = [Qt.AlignCenter] * 3 + [Qt.AlignVCenter | Qt.AlignLeft] * 2
        for label, position, alignment in zip(labels, positions, alignments):
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignCenter)
            left_display.addWidget(lbl, *position, alignment=alignment)
            
        # Model and ROI_auto images
        for folder, (row, col) in zip(["raw", "ccreg", "warped"], [(1,1), (1,2), (1,3)]):
            left_display.addWidget(self.models[folder], row, col, alignment=Qt.AlignCenter)
            
            left_display.addWidget(self.roi_autos[folder], row+1, col, alignment=Qt.AlignCenter)
            
        # Dropdown choice for last row
        combo = QComboBox()
        combo.setFocusPolicy(Qt.ClickFocus)
        combo.addItems(["segmentation", "rgb_init", "seg_init", "identities", 
                        "ΔR/R", "ΔF/F"])
        combo.setCurrentText("ΔR/R")
        self.changeChoice(combo.currentText())
        combo.activated[str].connect(self.changeChoice)
        left_display.addWidget(combo, 3, 0, alignment=Qt.AlignVCenter | Qt.AlignLeft)
        # Last row images
        for folder, position in zip(["raw", "ccreg", "warped"], [(3,1), (3,2), (3,3)]):
            left_display.addLayout(self.choice_layouts[folder], *position)
        
        # Bottom slider to change displayed frame
        nframe_hbox = QHBoxLayout()
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
        left_vbox.addLayout(nframe_hbox)
        
        
        ## Right layout with buttons and actions
        right_control = QVBoxLayout()
        
        button1 = QPushButton("Do nothing")
        button1.adjustSize()
        right_control.addWidget(button1)
        right_control.addStretch(1)
        button2 = QPushButton("Do nothing2")
        button2.adjustSize()
        right_control.addWidget(button2)
        
        top_hbox.addLayout(left_vbox)
        top_hbox.addLayout(right_control)
        
        self.setLayout(top_hbox)
    
    def changeFrame(self, num):
        """Change the displayed frame to given num."""
        self.nframe_edit.setText(str(num))
        self.nframe_sld.setSliderPosition(int(num))
        for folder in ["raw", "ccreg", "warped"]:
            self.roi_autos[folder].changeFrame(int(num))
            if isinstance(self.choice_widgets[folder], LabelImage):
                self.choice_widgets[folder].changeFrame(int(num))
    
    def changeChoice(self, choice):
        """Change the last display based on the ComboBox choice."""
        for folder in ["raw", "ccreg", "warped"]:
            self.choice_layouts[folder].removeWidget(self.choice_widgets[folder])
            self.choice_widgets[folder].close()
            
            path = os.path.join(self.experiment, CHOICE_PATHS[choice] % folder)
            image = io.imread(path)
            if choice in ["model", "identities"]: # apply identity colormap
                image = to_id_cmap(image, vmax=self.n_axons_max)
            
            # If traces plots, put the widget in a scroll area
            if choice in ["ΔR/R", "ΔF/F"]:
                self.choice_widgets[folder] = VerticalScrollImage(image)
            elif image.ndim == 2:
                self.choice_widgets[folder] = LabelImage(image)
            elif image.ndim == 3:
                if image.shape[-1] in [3, 4]: # assume RGB(A)
                    self.choice_widgets[folder] = LabelImage(image)
                else:
                    self.choice_widgets[folder] = LabelStack(image)
            else:
                    self.choice_widgets[folder] = LabelStack(image)
            
            
            self.choice_layouts[folder].addWidget(self.choice_widgets[folder],
                                                  alignment=Qt.AlignCenter)