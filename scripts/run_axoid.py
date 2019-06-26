#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script running AxoID on experiment(s), detecting and tracking ROIs, then
extracting fluorescence traces and saving outputs.
Created on Thu Jun  6 09:58:59 2019

@author: nicolas
"""

import os.path
import sys

# Add parent folder to path in order to access `axoid`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from axoid.main import main


if __name__ == "__main__":
    main()