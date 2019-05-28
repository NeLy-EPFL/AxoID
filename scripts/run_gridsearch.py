#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to launch a grid-search over hyperparamters, using the run_train.py script.
Created on Wed Oct 31 16:38:54 2018

@author: nicolas
"""

import os, sys, time
import numpy as np

# Add parent folder to path in order to access `axoid`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from axoid.utils.script import Arguments
from axoid.detection.deeplearning.model import CustomUNet
import run_train

# Parameters
n_epochs = 10
batch_sizes = [8, 16, 32]
learning_rates = [1e-4, 5e-4, 1e-3]
out1_channels = [8, 16]

def main():
    print("Starting on %s\n\nResults over validation data (%d epochs):\n" % (time.ctime(), n_epochs))
    start_time = time.time()
    
    # Arguments for run_train
    args = Arguments(
            batch_size = 32,
            crop_dice = False,
            data_aug = False,
            data_dir = "/data/talabot/datasets/datasets_190401/",
            epochs = n_epochs,
            eval_test = False,
            input_channels = "RG",
            learning_rate = 0.001,
            model_dir = None,
            no_gpu = False,
            pixel_weight = False,
            save_fig = False,
            scale_crop = 4.0,
            seed = 1,
            step_decay = None,
            synthetic_data = False,
            synthetic_only = False,
            synthetic_ratio = None,
            timeit = False,
            verbose = False
    )
    model = None
    args.pixel_weight = True
    print("Pixel weighting is enabled.")
    args.step_decay = 5
    print("Step decay every %d epochs." % args.step_decay)
    print()
    
    for bs in batch_sizes:
        args.batch_size = bs
        for lr in learning_rates:
            args.learning_rate = lr
            for out1_c in out1_channels:
                print("bs={: >2d} - lr={:.0E} - out1_c={: >2d}".format(bs, lr, out1_c), end="", flush=True)
                    
                try:
                    model = CustomUNet(len(args.input_channels), 
                                       u_depth = 4, out1_channels = out1_c, 
                                       batchnorm = True)
                    history = run_train.main(args, model=model)
                    best_epoch = np.argmax(history["val_dice"])
                    print(" | loss={:.6f} - dice={:.6f}".format(
                            history["val_loss"][best_epoch], history["val_dice"][best_epoch]), end="")
                    if args.crop_dice:
                        print(" - diC{:.1f}={:.6f}".format(
                                args.scale_crop, history["val_diC%.1f" % args.scale_crop][best_epoch]),
                        end="")
                    print()
                    
                except RuntimeError as err: # CUDA out of memory
                    print(" | RuntimeError ({})".format(err))
                            
    # If an error occured, this is not printed
    # TODO: is there a way to force this to print ? try-except does not work with KeyboardInterrupt
    duration = time.time() - start_time
    duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(
            duration // 3600, (duration // 60) % 60, duration % 60)
    print("\nEnding on %s" % time.ctime())
    print("Gridsearch duration: %s." % duration_msg)


if __name__ == "__main__":
    main()
