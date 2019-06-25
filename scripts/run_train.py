#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to train a network.
Created on Thu Nov  1 10:45:50 2018

@author: nicolas
"""

import os, sys, time, shutil
import argparse
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
from skimage import io
import imgaug.augmenters as iaa

import torch

# Add parent folder to path in order to access `axoid`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from axoid.detection.deeplearning.data import get_all_dataloaders, normalize_range, pad_transform
from axoid.detection.deeplearning.loss import get_BCEWithLogits_loss
from axoid.detection.deeplearning.metric import get_dice_metric, get_crop_dice_metric
from axoid.detection.deeplearning.model import CustomUNet
from axoid.detection.deeplearning.train import train, train_plot
from axoid.detection.deeplearning.test import evaluate

def main(args, model=None):
    """
    Main function of the run_train script, can be used as is with correct arguments (and optional model).
    
    Parameters
    ----------
    args : arguments
        Arguments passed to the script through the command line.
    model : PyTorch model (optional)
        PyTorch model to train. If not given, one will be made in the code.
    """
    ## Initialization
    if args.timeit:
        start_time = time.time()
    u_depth = 4
    out1_channels = 16
    
    # Seed the script
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed*10 + 1234)
    torch.manual_seed(seed*100 + 4321)
    
    # Device selection (note that the current code does not work with multi-GPU)
    if torch.cuda.is_available() and not args.no_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if args.verbose:
        print("Device set to '{}'.".format(device))
    
    # Create random augment sequence for data augmentation
    if args.data_aug:
        aug_seq = iaa.Sequential([
            iaa.GammaContrast((0.7, 1.3)) # Gamma correction
        ])
        aug_fn = aug_seq.augment_image
        if args.verbose:
            print("Data augmentation is enabled.")
    else:
        aug_fn = lambda x: x # identity function
    
    ## Data preparation    
    # Create dataloaders
    dataloaders = get_all_dataloaders(
        args.data_dir, 
        args.batch_size, 
        input_channels = args.input_channels, 
        test_dataloader = args.eval_test,
        use_weights = args.pixel_weight,
        synthetic_data = args.synthetic_data,
        synthetic_ratio = args.synthetic_ratio,
        synthetic_only = args.synthetic_only,
        train_transform = lambda img: normalize_range(pad_transform(aug_fn(img), u_depth)), 
        train_target_transform = lambda img: pad_transform(img, u_depth),
        eval_transform = lambda img: normalize_range(pad_transform(img, u_depth)), 
        eval_target_transform = lambda img: pad_transform(img, u_depth)
    )
    
    N_TRAIN = len(dataloaders["train"].dataset)
    N_VALID = len(dataloaders["valid"].dataset)
    if args.eval_test:
        N_TEST = len(dataloaders["test"].dataset)
    # Compute class weights (as pixel imbalance)
    pos_count = 0
    neg_count = 0
    for filename in dataloaders["train"].dataset.y_filenames:
        y = io.imread(filename)
        pos_count += (y == 255).sum()
        neg_count += (y == 0).sum()
    pos_weight = torch.tensor((neg_count + pos_count) / (2 * pos_count)).to(device)
    neg_weight = torch.tensor((neg_count + pos_count) / (2 * neg_count)).to(device)
    
    if args.verbose:
        print("%d train images" % N_TRAIN, end="")
        if args.synthetic_only:
            print(" (of synthetic data only).")
        elif args.synthetic_data:
            if args.synthetic_ratio is None:
                print(" (with synthetic data).")
            else:
                print(" (with %d%% of synthetic data)." % (args.synthetic_ratio * 100))
        else:
            print(".")
        print("%d validation images." % N_VALID)
        if args.eval_test:
            print("%d test images." % N_TEST)
        if args.pixel_weight:
            print("Pixel-wise weighting enabled.")
        print("{:.3f} positive weighting.".format(pos_weight.item()))
        print("{:.3f} negative weighting.".format(neg_weight.item()))
    
    ## Model, loss, metrics, and optimizer definition
    if model is None:
        model = CustomUNet(len(args.input_channels), u_depth=u_depth,
                           out1_channels=out1_channels, batchnorm=True, device=device)
        if args.model_dir is not None:
            # Save the "architecture" of the model by copy/pasting the class definition file
            os.makedirs(os.path.join(args.model_dir), exist_ok=True)
            shutil.copy("../axoid/detection/deeplearning/model.py", 
                        os.path.join(args.model_dir, "utils_model_save.py"))
    # Make sure the given model is on the correct device
    else: 
        model.to(device)
    if args.verbose:
        print("\nModel definition:", model, "\n")
    
    loss_fn = get_BCEWithLogits_loss(pos_weight=pos_weight, neg_weight=neg_weight)
    
    metrics = {"dice": get_dice_metric()}
    if args.crop_dice:
        metrics.update({"diC%.1f" % args.scale_crop: get_crop_dice_metric(scale=args.scale_crop)})
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.step_decay is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_decay, 0.5)
        if args.verbose:
            print("Learning rate decay is enabled with step = %d epochs." % args.step_decay)
    else:
        scheduler = None
    
    ## Train the model
    best_model, history = train(model,
                                dataloaders,
                                loss_fn,
                                optimizer,
                                args.epochs,
                                scheduler = scheduler,
                                metrics = metrics,
                                criterion_metric = "dice",
                                model_dir = args.model_dir,
                                replace_dir = True,
                                verbose = args.verbose)
    
    ## Save a figure if applicable
    if args.save_fig and args.model_dir is not None:
        fig = train_plot(history, crop_dice=args.crop_dice, scale_crop=args.scale_crop)
        fig.savefig(os.path.join(args.model_dir, "train_fig.png"), dpi=400)
        print("Training figure saved at %s." % os.path.join(args.model_dir, "train_fig.png"))
    if args.model_dir is not None:
        print("Best model saved under %s." % args.model_dir)
       
    ## Evaluate best model over test data
    if args.eval_test:
        test_metrics = evaluate(best_model, dataloaders["test"], 
                                {"loss": loss_fn, **metrics})
        if args.verbose:
            print("\nTest loss = {}".format(test_metrics["loss"]))
            print("Test dice = {}".format(test_metrics["dice"]))
            if args.crop_dice:
                print("Crop dice = {}".format(test_metrics["diC%.1f" % args.scale_crop]))
        
    ## Display script duration if applicable
    if args.timeit:
        duration = time.time() - start_time
        duration_msg = "{:.0f}h {:02.0f}min {:02.0f}s".format(duration // 3600, (duration // 60) % 60, duration % 60)
        print("\nScript took %s." % duration_msg)
        
    # If model was evaluated on test data, return the best metric values, and 
    # return in any case the history. This is useless in this script, but allow 
    # this function to be reused somewhere else, e.g. for the gridsearch.
    if args.eval_test:
        return history, test_metrics
    else:
        return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a network to detect ROI on single images.")
    parser.add_argument(
            '--batch_size', 
            type=int,
            default=16, 
            help="batch_size for the dataloaders (default=16)"
    )
    parser.add_argument(
            '--crop_dice', 
            action="store_true",
            help="enable the use of the cropped dice coefficient as a performance metric"
    )
    parser.add_argument(
            '--data_aug', 
            action="store_true",
            help="enable data augmentation on train set (see code for augmentation sequence)"
    )
    parser.add_argument(
            '--data_dir',
            type=str,
            default="/data/talabot/datasets/datasets_190510/", 
            help="directory to the train, validation, (test), and (synthtetic) data. "
            "It should contain train/, validation/, test/, and synthetic/ subdirs "
            "(test/ and synthetic/ are not mandatory, see --eval_test and --synthetic_*). "
            "These should be structured as: "
            "train_dir-->subdirs-->rgb_frames: folder with input images; and "
            "train_dir-->subdirs-->seg_frames: folder with target images; and (optional, see --pixel_weight)"
            "train_dir-->subdirs-->wgt_frames: folder with weight images "
            "(default=/data/talabot/datasets/datasets_190510/)"
    )
    parser.add_argument(
            '--epochs', 
            type=int,
            default=10, 
            help="number of epochs (default=10)"
    )
    parser.add_argument(
            '--eval_test', 
            action="store_true",
            help="perform a final evaluation over the test data"
    )
    parser.add_argument(
            '--input_channels', 
            type=str,
            default="RG", 
            help="channels of RGB input images to use (default=RG)"
    )
    parser.add_argument(
            '--learning_rate', 
            type=float,
            default=0.0005,
            help="learning rate for the stochastic gradient descent (default=0.0005)"
    )
    parser.add_argument(
            '--model_dir', 
            type=str,
            help="directory where the model is to be saved (if not set, the model won't be saved)"
    )
    parser.add_argument(
            '--no_gpu', 
            action="store_true",
            help="disable gpu utilization (not needed if no gpu are available)"
    )
    parser.add_argument(
            '--pixel_weight', 
            action="store_true",
            help="enable pixel-wise weighting using pre-computed weight images"
    )
    parser.add_argument(
            '--save_fig', 
            action="store_true",
            help="save a figure of the training loss and metrics with the model "
            "(requires the --model_dir argument to be set)"
    )
    parser.add_argument(
            '--scale_crop', 
            type=float,
            default=4.0,
            help="scaling of the cropping (w.r.t. ROI's bounding box) for "
            "the cropped metrics. (default=4.0)"
    )
    parser.add_argument(
            '--seed', 
            type=int,
            default=1,
            help="initial seed for RNG (default=1)"
    )
    parser.add_argument(
            '--step_decay', 
            type=int,
            default=None,
            help="number of epochs after which the learning rate is decayed by "
            "a factor 2. If not set, not decay is used."
    )
    parser.add_argument(
            '--synthetic_data', 
            action="store_true",
            help="enable the use of synthetic data for training"
    )
    parser.add_argument(
            '--synthetic_only', 
            action="store_true",
            help="use only the synthetic data for training. This is different "
            "than using --synthetic_ration 1.0 as it will always use all the "
            "synthetic data, without being limited to the number of frames "
            "in the train folder"
    )
    parser.add_argument(
            '--synthetic_ratio', 
            type=float,
            default=None,
            help="(requires --synthetic_data) ratio of synthetic data "
            "vs. real data. If not set, all real and synthetic data are used"
    )
    parser.add_argument(
            '-t', '--timeit', 
            action="store_true",
            help="time the script"
    )
    parser.add_argument(
            '-v', '--verbose', 
            action="store_true",
            help="enable output verbosity"
    )
    args = parser.parse_args()
    
    main(args)
