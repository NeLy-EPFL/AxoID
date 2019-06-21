#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the model correction window widget.
This GUI page allows the user to modify the model of the tracker to improve the 
axon's identities (e.g.: by fusing, cutting, or discarding ROIs).
Created on Wed Jun 19 11:48:44 2019

@author: nicolas
"""

import os
import pickle
import shutil

import numpy as np
from skimage import io
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QLabel, QHBoxLayout, QVBoxLayout, QComboBox, QGroupBox,
                             QGridLayout, QPushButton, QRadioButton, QButtonGroup,
                             QAbstractButton)

from .constants import PAGE_CORRECTION, CHOICE_PATHS, ID_CMAP
from .image import LabelImage, LabelStack, VerticalScrollImage
from .multipage import AxoidPage
from axoid.utils.image import to_id_cmap


class ModelPage(AxoidPage):
    """Page of the model correction process."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the model page."""
        super().__init__(*args, **kwargs)
        
        # Get max number of axons amongst outputs
        with open(os.path.join(self.experiment, "output", "GC6_auto",
                               "final", "dRR_dic.p"), "rb") as f:
            self.n_axons_max = len(pickle.load(f).keys())
        
        # Create intermediate gui folder
        for folder in ["axoid_internal", "GC6_auto", "ROI_auto"]:
            gui_path = os.path.join(self.experiment, "output", folder, "gui")
            if os.path.isdir(gui_path):
                shutil.rmtree(gui_path)
            shutil.copytree(os.path.join(self.experiment, "output", folder, "final"),
                            gui_path)
        
        # Initialize images and stacks
        model_path = os.path.join(self.experiment, "output", "axoid_internal",
                                  "final", "model.tif")
        model_img = to_id_cmap(io.imread(model_path),
                               cmap=ID_CMAP, vmax=self.n_axons_max)
        self.model = LabelImage(model_img)
        self.model_new = LabelImage(model_img)
        
        identities_path = os.path.join(self.experiment, "output", "axoid_internal",
                                       "final", "identities.tif")
        identities_img = to_id_cmap(io.imread(identities_path),
                                    cmap=ID_CMAP, vmax=self.n_axons_max)
        self.identities = LabelStack(identities_img)
        self.identities_new = LabelStack(identities_img)
        for folder in ["final", "gui"]:            
            self.choice_layouts.update({folder: QHBoxLayout()})
            self.choice_widgets.update({folder: QLabel()})
            self.choice_layouts[folder].addWidget(self.choice_widgets[folder],
                                                  alignment=Qt.AlignCenter)
        
        self.initUI()
    
    def initUI(self):
        """Initialize the interface."""
        ## Left display
        for row in [1, 2]:
            self.left_display.setRowStretch(row, 1)
        for col in [1, 2, 3]:
            self.left_display.setColumnStretch(col, 1)
        
         # Row and column labels
        labels = ["Model", "Identities", "In folder", "Current"]
        positions = [(0,1), (0,2), (1,0), (2,0)]
        alignments = [Qt.AlignCenter] * 2 + [Qt.AlignVCenter | Qt.AlignLeft] * 2
        for label, position, alignment in zip(labels, positions, alignments):
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignCenter)
            self.left_display.addWidget(lbl, *position, alignment=alignment)
        
        # Model and identities images
        self.left_display.addWidget(self.model, 1, 1, alignment=Qt.AlignCenter)
        self.left_display.addWidget(self.model_new, 2, 1, alignment=Qt.AlignCenter)
        
        self.left_display.addWidget(self.identities, 1, 2, alignment=Qt.AlignCenter)
        self.left_display.addWidget(self.identities_new, 2, 2, alignment=Qt.AlignCenter)
        
        # Dropdown choice for last column
        combo = QComboBox()
        combo.setFocusPolicy(Qt.ClickFocus)
        combo.addItems(["segmentation", "rgb_init", "seg_init", "ROI_auto", 
                        "ΔR/R", "ΔF/F"])
        combo.setCurrentText("rgb_init")
        self.changeChoice(combo.currentText())
        combo.activated[str].connect(self.changeChoice)
        self.left_display.addWidget(combo, 0, 3, alignment=Qt.AlignCenter)
        # Last column images
        for folder, position in zip(["final", "gui"], [(1,3), (2,3)]):
            self.left_display.addLayout(self.choice_layouts[folder], *position)
        
        ## Right control with buttons and actions
        # Toolbox group
        tool_vbox = QVBoxLayout()
        tool_groupbox = QGroupBox("Tools")
        self.tool_group = QButtonGroup()
        tool_layout = QGridLayout()
        for i, position in enumerate([(0,0), (0,1), (1,0), (1,1)]):
            button = QPushButton("Tool %d" % i)
            button.setCheckable(True)
            tool_layout.addWidget(button, *position)
            self.tool_group.addButton(button)
        tool_groupbox.setLayout(tool_layout)
        self.tool_group.buttonClicked[QAbstractButton].connect(self.changeTool)
        tool_vbox.addWidget(tool_groupbox, alignment=Qt.AlignCenter)
        tool_vbox.addStretch(1)
        
        self.tool_actions = QVBoxLayout()
        self.tool_actions_widget = QLabel()
        self.tool_actions.addWidget(self.tool_actions_widget)
        self.tool_actions.addStretch(1)
        
        # Apply/save/etc.
        fn_vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.setToolTip("Apply changes made to model to current results")
        save_btn = QPushButton("Save")
        save_btn.setToolTip("Save current results in the final/ folder")
        hbox.addWidget(apply_btn)
        hbox.addWidget(save_btn)
        finish_btn = QPushButton("Finish")
        finish_btn.setToolTip("Finish AxoID's user correction GUI with current results")
        correction_btn = QPushButton("Frame correction")
        correction_btn.setToolTip("Finish model correction and move on to frame correction")
        
        fn_vbox.addLayout(hbox)
        fn_vbox.addWidget(finish_btn)
        fn_vbox.addWidget(correction_btn)
        fn_vbox.addStretch(1)
        
        self.right_control.addStretch(1)
        self.right_control.addLayout(tool_vbox, stretch=1.5)
        self.right_control.addLayout(self.tool_actions, stretch=5)
        self.right_control.addLayout(fn_vbox, stretch=2)
    
    def changeFrame(self, num):
        """Change the displayed frames to the given num."""
        super().changeFrame(num)
        self.identities.changeFrame(int(num))
        self.identities_new.changeFrame(int(num))
        for folder in ["final", "gui"]:
            if isinstance(self.choice_widgets[folder], LabelImage):
                self.choice_widgets[folder].changeFrame(int(num))
    
    def changeChoice(self, choice):
        """Change the last display based on the ComboBox choice."""
        for folder in ["final", "gui"]:
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
            else:
                    self.choice_widgets[folder] = LabelStack(image)
            
            
            self.choice_layouts[folder].addWidget(self.choice_widgets[folder],
                                                  alignment=Qt.AlignCenter)
    
    def changeTool(self, tool_btn):
        """Change the available actions based on the selected tool."""
        tool_groupbox = QGroupBox(tool_btn.text())
        tool_groupbox.setAlignment(Qt.AlignCenter)
        tool_group = QButtonGroup()
        tool_layout = QVBoxLayout()
        for i in range(int(tool_btn.text()[-1])):
            button = QPushButton("%s: Tool %d" % (tool_btn.text(), i))
            button.setCheckable(True)
            tool_layout.addWidget(button)
            tool_group.addButton(button)
        tool_groupbox.setLayout(tool_layout)
        self.tool_actions.replaceWidget(self.tool_actions_widget, tool_groupbox)
        self.tool_actions_widget.close()
        self.tool_actions_widget = tool_groupbox