#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing common constants for the GUI.
Created on Thu Jun 20 13:16:31 2019

@author: nicolas
"""

import os.path

## Constants
# Pages identifiers
PAGE_SELECTION = 1      # output selection between raw/ccreg/warped
PAGE_MODEL = 2          # correction of the model
PAGE_CORRECTION = 3     # frame-wise correction
PAGE_ANNOTATION = 4     # manual annotations
# Paths to different data for display choices
CHOICE_PATHS = {#"AVG_RGB": os.path.join("2Pimg"),
                #"AVG_tdTom": os.path.join("2Pimg"),
                #"input": os.path.join("2Pimg"),
                "segmentation": os.path.join("output", "axoid_internal", "%s", "segmentations.tif"),
                "rgb_init": os.path.join("output", "axoid_internal", "%s", "rgb_init.tif"),
                "seg_init": os.path.join("output", "axoid_internal", "%s", "seg_init.tif"),
                "model": os.path.join("output", "axoid_internal", "%s", "model.tif"),
                "identities": os.path.join("output", "axoid_internal", "%s", "identities.tif"),
                "ROI_auto": os.path.join("output", "ROI_auto", "%s", "RGB_seg.tif"),
                "ΔR/R": os.path.join("output", "GC6_auto", "%s", "ROIs_dRR.png"),
                "ΔF/F": os.path.join("output", "GC6_auto", "%s", "ROIs_dFF.png")}