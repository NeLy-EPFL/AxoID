#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module of AxoID user correction GUI with the main window widget.
Created on Tue Jun 18 17:16:11 2019

@author: nicolas
"""

import os.path
import sys
import shutil
import argparse

from PyQt5.QtWidgets import QApplication, QHBoxLayout

from .multipage import MultiPageWidget
from .selection import SelectionPage
from .model import ModelPage
from .correction import CorrectionPage
from .annotation import AnnotationPage
from .constants import PAGE_SELECTION, PAGE_MODEL, PAGE_CORRECTION, PAGE_ANNOTATION


class AxoIDWindow(MultiPageWidget):
    """
    Main window for the AxoID GUI.
    
    It allows navigation through multiple pages:
        1. Output selection: the user select which output amongst raw, ccreg,
           and warped he wants to use.
        2. Model correction: the user can modify the tracker model to discard,
           fuse ROIs, or apply "cuts" to separate touching axons. Then he can apply
           theses changes to the whole experiment automatically.
        3. Frame correction: the user can modify single frames, where the 
           binary detection or the tracking is erroneous.
        4. Manual annotation: (not implemented) the user can choose to discard
           all the outputs, and manually annotated frames to fine tune the 
           detector network, and then apply the rest of the pipeline.
    """
    
    def __init__(self, experiment, *args, goto=None, **kwargs):
        """
        Initialize the window and its pages.
        
        Parameters
        ----------
        experiment : str
            Path to the experiment folder (excluding 2Pimg/ or output/).
        args : list of arguments
            List of arguments to pass to MultiPageWidget initialization.
        goto : int (optional)
            Page identifier (see constants.py). If set, the GUI will be started
            on this page instead of the output selection one.
        kwargs : dict of named arguments
            Dictionary of named arguments to pass to MultiPageWidget initialization.
        """
        super().__init__(*args, **kwargs)
        
        self.experiment = experiment
        
        # Set pages
        self.addPage(SelectionPage(self.experiment), PAGE_SELECTION)
        self.addPage(ModelPage(self.experiment), PAGE_MODEL)
        self.addPage(CorrectionPage(self.experiment), PAGE_CORRECTION)
        self.addPage(AnnotationPage(self.experiment), PAGE_ANNOTATION)
        
        self.pageWidgets[PAGE_SELECTION].initUI()
        self.initUI()
        
        # Change page
        if goto is not None:
            self.changePage(goto)
    
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
    
    def closeEvent(self, event):
        """Clean before quitting the GUI."""
        # Delete the gui/ result folders
        for folder in ["axoid_internal", "GC6_auto", "ROI_auto"]:
            gui_path = os.path.join(self.experiment, "output", folder, "gui")
            if os.path.isdir(gui_path):
                shutil.rmtree(gui_path)


def main(args=None):
    """Initialize and start the GUI."""
    if args is None:
        args = parser()
    
    # Go to specific pages
    if args.model:
        if not os.path.isdir(os.path.join(args.experiment, "output",
                                          "axoid_internal", "final")):
            raise RuntimeError("cannot start on model correction page without final outputs")
        goto = PAGE_MODEL
    elif args.correction:
        if not os.path.isdir(os.path.join(args.experiment, "output",
                                          "axoid_internal", "final")):
            raise RuntimeError("cannot start on frame correction page without final outputs")
        goto = PAGE_CORRECTION
    else:
        goto = None
    
    app = QApplication([])
    
    window = AxoIDWindow(args.experiment, goto=goto)
    
    # Set the window to 0.9 * screen dimension while preserving its ratio
    screen = app.primaryScreen()
    size = screen.size()
    width, height = size.width(), size.height()
    app_width = width * 0.9
    app_height = height * 0.9
    app_width = min(app_width, window.wh_ratio * app_height)
    app_height = min(app_height, window.wh_ratio * app_width)
    window.resize(app_width, app_height)
    window.show()
    
    sys.exit(app.exec_())


def parser():
    """
    Parse the command for arguments.
    
    Returns
    -------
    args : arguments
        Arguments passed to the script through the command line.
    """
    parser = argparse.ArgumentParser(
            description="User correction GUI of AxoID.")
    parser.add_argument(
            'experiment',
            type=str,
            help="path to the experiment folder (excluding \"2Pimg/\")"
    )
    parser.add_argument(
            '--model',
            action="store_true",
            help="start the GUI on the model correction page"
    )
    parser.add_argument(
            '--correction',
            action="store_true",
            help="start the GUI on the frame correction page"
    )
    args = parser.parse_args()
    
    return args