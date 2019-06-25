#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate the synthetic dataset for the training.
See synthetic_generation.ipynb for more details.

Created on Thu Nov 22 10:01:56 2018

@author: nicolas
"""

import os, sys, time
import warnings

import numpy as np
from skimage import io

# Add parent folder to path in order to access `axoid`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from axoid.utils.image import to_npint
from axoid.detection.synthetic.generation import synthetic_stack


if __name__ == "__main__":
    ## Parameters and constants (shape and n_neurons are below)
    n_stacks = 4
    n_images = 20
    
    date = time.strftime("%y%m%d", time.localtime())
    synth_dir = "/data/talabot/datasets/datasets_190510/synthetic_%s/" % date
    
    start = time.time()
    for i in range(n_stacks):
        ### Random parameters here ##
        # Randomized shape
        if np.random.rand() < 0.5: # square image half of the time
            rand_size = np.random.randint(6, 10 + 1) * 32
            shape = (rand_size, rand_size)
        else:
            rand_h = np.random.randint(6, 10 + 1) * 32
            rand_w = np.random.randint(rand_h/32, 10 + 1) * 32
            shape = (rand_h, rand_w)
        # Randomized n_neurons
        n_neurons = np.random.randint(2, 6 + 1)
        
        folder = os.path.join(synth_dir, "synth_{}neur_{:03d}".format(n_neurons, i))
        print("Creating stack %d/%d" % (i + 1, n_stacks), end="")
        print("  - folder:", folder)
        
        synth_stack, synth_seg = synthetic_stack(shape, n_images, n_neurons, 
                                                 cyan_gcamp=True, return_label=True)
        
        os.makedirs(folder, exist_ok=True)
        os.makedirs(os.path.join(folder, "rgb_frames"), exist_ok=True)
        os.makedirs(os.path.join(folder, "seg_frames"), exist_ok=True)
        os.makedirs(os.path.join(folder, "lbl_frames"), exist_ok=True)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Save full stacks
            io.imsave(os.path.join(folder, "RGB.tif"), to_npint(synth_stack))
            io.imsave(os.path.join(folder, "seg_ROI.tif"), to_npint(synth_seg.astype(np.bool)))
            io.imsave(os.path.join(folder, "lbl_ROI.tif"), to_npint(synth_seg))
            # Save image per image
            for j in range(n_images):
                io.imsave(os.path.join(folder, "rgb_frames", "rgb_{:04}.png".format(j)), to_npint(synth_stack[j]))
                io.imsave(os.path.join(folder, "seg_frames", "seg_{:04}.png".format(j)), to_npint(synth_seg[j].astype(np.bool)))
                io.imsave(os.path.join(folder, "lbl_frames", "lbl_{:04}.png".format(j)), to_npint(synth_seg[j]))
    
    duration = time.time() - start
    print("\nScript took {:02.0f}min {:02.0f}s.".format(duration // 60, duration % 60))
    
    # Launch the mask generation over the newly created synthetic dataset
    print("Launching weight generation script...")
    dir_path = os.path.dirname(__file__)
    os.system("python %s --data_dir %s --separation_border" % \
              (os.path.join(dir_path, "..", "scripts", "generate_weights.py"),
               synth_dir))