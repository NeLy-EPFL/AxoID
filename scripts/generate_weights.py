#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script used to generate weights for pixel-wise loss weighting.
Weights are rescaled at train time by weight = neg_w + (pos_w - neg_w) * weight,
so the weights computed here should simply be the relative values to negative and
positive weighting that are computed before the training.
neg_w : corresponds to the weight of the negative class (i.e. background)
pos_w : corresponds to the weight of the positive class (i.e. ROIs)

Created on Wed Feb 20 16:57:32 2019

@author: nicolas
"""

import os, sys, warnings, argparse
import numpy as np
from skimage import io

# Add parent folder to path in order to access `axoid`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from axoid.utils.image import imread_to_float, to_npint
from axoid.detection.deeplearning.data import compute_weights

def main(args):
    """Generate the weight images."""
    if args.data_dir is None:
        data_dir = "/data/talabot/datasets/datasets_190401_sep"
        sets = ["train", "validation", "test", "synthetic_190401"]
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
            weights = compute_weights(seg_stack, contour=True, separation=args.separation_border)
                
            # Save results
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(os.path.join(data_dir, set, exp, "weights.tif"), 
                          to_npint(weights, dtype=np.uint16, float_scaling=255))
                os.makedirs(os.path.join(data_dir, set, exp, "wgt_frames"), exist_ok=True)
                for j in range(len(weights)):
                    io.imsave(os.path.join(data_dir, set, exp, "wgt_frames", "wgt_%04d.png" % j), 
                              to_npint(weights[j], dtype=np.uint16, float_scaling=255))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate weights for pixel-wise"
                                     "loss weighting.")
    parser.add_argument(
            '--data_dir',
            type=str, 
            help="directory to the experiments folder. If not set, will use the"
            "one in the code."
    )
    parser.add_argument(
            '--separation_border',
            action="store_true",
            help="add the weights for separation between close ROIs.")
    args = parser.parse_args()
    
    main(args)