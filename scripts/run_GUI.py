#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to run the AxoID GUI.
Created on Wed Jun 19 11:39:58 2019

@author: nicolas
"""

import os.path
import sys

# Add parent folder to path in order to access `axoid`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from axoid.GUI.main import main


if __name__ == "__main__":
    main()