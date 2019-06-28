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
import pickle
import shutil

import numpy as np
from matplotlib import cm, colorbar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from skimage import io
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton

from .constants import PAGE_MODEL, CHOICE_PATHS, ID_CMAP
from .image import LabelImage, LabelStack, VerticalScrollImage
from .multipage import AxoidPage
from axoid.utils.image import to_id_cmap


class SelectionPage(AxoidPage):
    """
    Page of the output selection process.
    
    Here, the user select which output amongst raw data, cross-correlation 
    registered data, and optic flow warped data. He can explore the outputs of
    the three results before making a choice. If an output was already selected,
    he can also choose to continue with it instead of starting again.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the selection page.
        
        Args and kwargs are passed to the AxoidPage constructor.
        """
        super().__init__(*args, **kwargs)
    
    def initUI(self):
        """Initialize the interface."""
        # Get max number of axons amongst outputs
        self.n_axons_max = -np.inf
        for folder in ["raw", "ccreg", "warped"]:
            with open(os.path.join(self.experiment, "output", "GC6_auto",
                                   folder, "dRR_dic.p"), "rb") as f:
                n_axons = len(pickle.load(f).keys())
            if n_axons > self.n_axons_max:
                self.n_axons_max = n_axons
        
        # Initialize images and stacks
        self.models = dict()
        self.rois = dict()
        self.choice_layouts = dict()
        self.choice_widgets = dict()
        for folder in ["raw", "ccreg", "warped"]:
            model_path = os.path.join(self.experiment, "output", "axoid_internal",
                                      folder, "model.tif")
            model_img = to_id_cmap(io.imread(model_path), cmap=ID_CMAP, vmax=self.n_axons_max)
            model_lbl = LabelImage(model_img)
            self.models.update({folder: model_lbl})
            
            roi_path = os.path.join(self.experiment, "output", "ROI_auto",
                                    folder, "RGB_seg.tif")
            roi_img = io.imread(roi_path)
            roi_lbl = LabelStack(roi_img)
            self.rois.update({folder: roi_lbl})
            
            self.choice_layouts.update({folder: QHBoxLayout()})
            self.choice_widgets.update({folder: QLabel()})
            self.choice_layouts[folder].addWidget(self.choice_widgets[folder],
                                                  alignment=Qt.AlignCenter)
        # Get colorbar image
        fig, ax = plt.subplots(1, 1, figsize=(0.5,3),
                               facecolor=[1,1,1,0], constrained_layout=True)
        cb = colorbar.ColorbarBase(ax, cmap=cm.get_cmap(ID_CMAP))
        cb.ax.invert_yaxis()
        cb.set_ticks(np.arange(self.n_axons_max) / (self.n_axons_max - 1))
        cb.set_ticklabels(np.arange(self.n_axons_max))
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background-color:transparent;")
        
        ## Left display
        for row in [1, 2, 3]:
            self.left_display.setRowStretch(row, 1)
        for col in [1, 2, 3]:
            self.left_display.setColumnStretch(col, 1)
            
        # Row and column labels
        labels = ["Raw", "CCReg", "Warped", "Model", "ROIs"]
        positions = [(0,1), (0,2), (0,3), (1,0), (2,0)]
        alignments = [Qt.AlignCenter] * 3 + [Qt.AlignVCenter | Qt.AlignLeft] * 2
        for label, position, alignment in zip(labels, positions, alignments):
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignCenter)
            self.left_display.addWidget(lbl, *position, alignment=alignment)
            
        # Model and ROIs images
        for folder, (row, col) in zip(["raw", "ccreg", "warped"], [(1,1), (1,2), (1,3)]):
            self.left_display.addWidget(self.models[folder], row, col, alignment=Qt.AlignCenter)
            
            self.left_display.addWidget(self.rois[folder], row+1, col, alignment=Qt.AlignCenter)
        
        # Dropdown choice for last row
        combo = QComboBox()
        combo.setFocusPolicy(Qt.ClickFocus)
        combo.addItems(["segmentation", "rgb_init", "seg_init", "identities", 
                        "ΔR/R", "ΔF/F"])
        combo.setCurrentText("ΔR/R")
        self.changeChoice(combo.currentText())
        combo.activated[str].connect(self.changeChoice)
        self.left_display.addWidget(combo, 3, 0, alignment=Qt.AlignVCenter | Qt.AlignLeft)
        # Last row images
        for folder, position in zip(["raw", "ccreg", "warped"], [(3,1), (3,2), (3,3)]):
            self.left_display.addLayout(self.choice_layouts[folder], *position)
        
        ## Right control with buttons and actions
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.canvas)
        hbox.addStretch(1)
        
        self.select_combo = QComboBox()
        self.select_combo.setFocusPolicy(Qt.ClickFocus)
        self.select_combo.addItems(["Choose output", "raw", "ccreg", "warped"])
        if os.path.isdir(os.path.join(self.experiment, "output", "axoid_internal", "final")):
            self.select_combo.addItems(["final"])
        self.select_combo.setCurrentText("Choose output")
        self.select_combo.activated[str].connect(self.selectOutput)
        self.select_btn = QPushButton("Select output")
        self.select_btn.setToolTip("Select chosen output as final output")
        self.select_btn.clicked.connect(self.saveFinalOutput)
        self.select_btn.setEnabled(False)
        self.annotation_btn = QPushButton("Manual annotation")
        self.annotation_btn.setToolTip("Manual annotations are not implemented yet")
        self.annotation_btn.setEnabled(False)
        
        self.right_title.setText("<b>Output selection</b>")
        self.right_control.addWidget(QLabel("Axon identity"), alignment=Qt.AlignCenter)
        self.right_control.addLayout(hbox)
        self.right_control.addStretch(5)
        self.right_control.addWidget(self.select_combo)
        self.right_control.addWidget(self.select_btn)
        self.right_control.addWidget(self.annotation_btn)
        self.right_control.addStretch(1)
    
    def changeFrame(self, num):
        """
        Change the stack frames to display.
        
        Parameters
        ----------
        num : int
            Index of the frame in the stacks to display.
        """
        super().changeFrame(num)
        for folder in ["raw", "ccreg", "warped"]:
            self.rois[folder].changeFrame(int(num))
            if isinstance(self.choice_widgets[folder], LabelImage):
                self.choice_widgets[folder].changeFrame(int(num))
    
    def changeChoice(self, choice):
        """
        Change the choice display based on the ComboBox choice.
        
        Parameters
        ----------
        choice : str
            String corresponding to the user choice. See constants.py for a list
            of possible strings, and their corresponding files.
        """
        for folder in ["raw", "ccreg", "warped"]:
            self.choice_layouts[folder].removeWidget(self.choice_widgets[folder])
            self.choice_widgets[folder].close()
            
            path = os.path.join(self.experiment, CHOICE_PATHS[choice] % folder)
            image = io.imread(path)
            if choice in ["model", "identities"]: # apply identity colormap
                image = to_id_cmap(image, cmap=ID_CMAP, vmax=self.n_axons_max)
            
            if choice in ["ΔR/R", "ΔF/F"]:
                self.choice_widgets[folder] = VerticalScrollImage(image)
            elif image.ndim == 2:
                self.choice_widgets[folder] = LabelImage(image)
            elif image.ndim == 3:
                if image.shape[-1] in [3, 4]: # assume RGB(A)
                    self.choice_widgets[folder] = LabelImage(image)
                else:
                    self.choice_widgets[folder] = LabelStack(image)
                    self.choice_widgets[folder].changeFrame(self.nframe_sld.value())
            else:
                    self.choice_widgets[folder] = LabelStack(image)
                    self.choice_widgets[folder].changeFrame(self.nframe_sld.value())
            
            
            self.choice_layouts[folder].addWidget(self.choice_widgets[folder],
                                                  alignment=Qt.AlignCenter)
    
    def selectOutput(self, selection):
        """
        Enable or disable buttons based on output selection.
        
        Parameters
        ----------
        selection : str
            Name of the output selection in the QComboBox.
        """
        if selection == "Choose output":
            self.select_btn.setEnabled(False)
            self.annotation_btn.setEnabled(False)
        elif selection == "final":
            self.select_btn.setEnabled(True)
            self.annotation_btn.setEnabled(False)
        else:
            self.select_btn.setEnabled(True)
            self.annotation_btn.setEnabled(False)  # not available as long as annotation is not implemented
    
    def saveFinalOutput(self):
        """Save the selected output as the final output."""
        selection = self.select_combo.currentText()
        if selection != "final":
            for folder in ["axoid_internal", "GC6_auto", "ROI_auto"]:
                destination = os.path.join(self.experiment, "output", folder, "final")
                if os.path.isdir(destination):
                    shutil.rmtree(destination)
                shutil.copytree(os.path.join(self.experiment, "output", folder, selection),
                                destination)
        # End of output selection, go to model correction page
        self.changedPage.emit(PAGE_MODEL)