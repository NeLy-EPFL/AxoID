#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to generate weights for pixel-wise loss weighting.
Weights are rescaled at train time by weight = neg_w + (pos_w - neg_w) * weight,
so the weights computed here should simply be the relative values to negative and
positive weighting that are computed before the training.

Created on Wed Feb 20 16:57:32 2019

@author: nicolas
"""

import os, warnings, argparse
import numpy as np
import scipy.ndimage as ndi
from skimage import io

from utils_common.image import imread_to_float, to_npint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weights for pixel-wise"
                                     "loss weighting.")
    parser.add_argument(
            '--data_dir',
            type=str, 
            help="directory to the experiments folder. If not set, will use the"
            "one in the code."
    )
    args = parser.parse_args()
    
    if args.data_dir is None:
        data_dir = "/data/talabot/pdm/dataset/"
        sets = ["train", "validation", "test", "synthetic_2-6_181205"]
    else:
        data_dir = args.data_dir
        sets = [""]
    
    for set in sets:
        # Pass the synthetic folder (which is a soft link)
        if set == "synthetic":
            continue
        print("Processing", set)
        
        # Loop over experiments
        exp_list = sorted(os.listdir(os.path.join(data_dir, set)))
        for i, exp in enumerate(exp_list):
            print("  %d/%d" % (i + 1, len(exp_list)))
            
            seg_stack = imread_to_float(os.path.join(data_dir, set, exp, "seg_ROI.tif"))
            weights = np.zeros_like(seg_stack)
            
            # Loop over frames
            for j in range(len(seg_stack)):
                distances = ndi.distance_transform_edt(1 - seg_stack[j])
                weights[j] = np.exp(- (distances / 3.0) ** 2) - seg_stack[j]
                
            # Save results
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(os.path.join(data_dir, set, exp, "weights.tif"), to_npint(weights))
                os.makedirs(os.path.join(data_dir, set, exp, "wgt_frames"), exist_ok=True)
                for j in range(len(weights)):
                    io.imsave(os.path.join(data_dir, set, exp, "wgt_frames", "wgt_%04d.png" % j), 
                              to_npint(weights[j]))
