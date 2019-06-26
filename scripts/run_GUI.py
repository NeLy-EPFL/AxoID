#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run the AxoID GUI.
Created on Wed Jun 19 11:39:58 2019

@author: nicolas
"""

import os.path
import sys
import argparse

from PyQt5.QtWidgets import QApplication

# Add parent folder to path in order to access `axoid`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from axoid.GUI.constants import PAGE_MODEL
from axoid.GUI.mainwindow import AxoIDWindow

def main(args):
    """Initialize and start the GUI."""
    args = parser()
    
    # Go to specific pages
    if args.model:
        if not os.path.isdir(os.path.join(args.experiment, "output",
                                          "axoid_internal", "final")):
            raise RuntimeError("cannot start on model correction page without final outputs")
        goto = PAGE_MODEL
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
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    main()
