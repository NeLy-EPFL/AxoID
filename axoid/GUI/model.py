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
        self.tool_buttongroup = QButtonGroup()
        # General toolbox
        general_vbox = QVBoxLayout()
        general_subbox = QHBoxLayout()
        general_groupbox = QGroupBox("General")
        reset_btn = QPushButton("Reset model")
        reset_btn.setToolTip("Reset the model to the one in the final/ folder")
        undo_btn = QPushButton("Undo")
        undo_btn.setToolTip("Undo the last action")
        undo_btn.setShortcut("Ctrl+Z")
        for button in [reset_btn, undo_btn]:
            general_subbox.addWidget(button)
        general_groupbox.setLayout(general_subbox)
        general_vbox.addWidget(general_groupbox, alignment=Qt.AlignCenter)
        general_vbox.addStretch(1)
        # Identity modifications
        ids_vbox = QVBoxLayout()
        ids_subbox = QHBoxLayout()
        ids_groupbox = QGroupBox("Identities")
        fuse_btn = QPushButton("Fuse")
        fuse_btn.setToolTip("Select two identities and fuse them into a single one")
        discard_btn = QPushButton("Discard")
        discard_btn.setToolTip("Discard (erase) the identity")
        for button in [fuse_btn, discard_btn]:
            button.setCheckable(True)
            ids_subbox.addWidget(button)
            self.tool_buttongroup.addButton(button)
        ids_groupbox.setLayout(ids_subbox)
        ids_vbox.addWidget(ids_groupbox, alignment=Qt.AlignCenter)
        ids_vbox.addStretch(1)
        # ROI cutting
        cuts_vbox = QVBoxLayout()
        cuts_subbox = QHBoxLayout()
        cuts_groupbox = QGroupBox("Cutting")
        drawcut_btn = QPushButton("Draw cut")
        drawcut_btn.setToolTip("Draw a line representing a cut")
        applycut_btn = QPushButton("Apply cut")
        applycut_btn.setToolTip("Apply the drawn cut to the model")
        for button in [drawcut_btn, applycut_btn]:
            button.setCheckable(True)
            cuts_subbox.addWidget(button)
            self.tool_buttongroup.addButton(button)
        cuts_groupbox.setLayout(cuts_subbox)
        cuts_vbox.addWidget(cuts_groupbox, alignment=Qt.AlignCenter)
        cuts_vbox.addStretch(1)
        
        
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
        self.right_control.addLayout(general_vbox, stretch=1.5)
        self.right_control.addLayout(ids_vbox, stretch=1.5)
        self.right_control.addLayout(cuts_vbox, stretch=5)
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