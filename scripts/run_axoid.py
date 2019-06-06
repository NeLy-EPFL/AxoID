#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script running AxoID on experiment(s), detecting and tracking ROIs, then
extracting fluorescence traces and saving outputs.
Created on Thu Jun  6 09:58:59 2019

@author: nicolas
"""

import os, os.path, sys
import warnings
import shutil
import pickle
import time
import argparse
import random

import numpy as np
from skimage import filters, io, morphology
import torch

# Add parent folder to path in order to access `axoid`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from axoid.detection.clustering import similar_frames, segment_projection
from axoid.detection.deeplearning.model import load_model
from axoid.detection.deeplearning.data import (compute_weights, normalize_range, 
                                               pad_transform_stack)
from axoid.detection.deeplearning.finetuning import fine_tune
from axoid.detection.deeplearning.test import predict_stack
from axoid.tracking.model import InternalModel
from axoid.utils.image import imread_to_float, to_npint
from axoid.utils.fluorescence import get_fluorophores, compute_fluorescence


## Constants
# Deep learning
LEARNING_RATE = 0.0005
BATCH_SIZE = 16
U_DEPTH = 4
OUT1_CHANNELS = 16
MIN_AREA = 11  # minimum size of ROI in pixels
# Fluorescence extraction
BIN_S = 10.0  # bin length for baseline computation (s)
RATE_HZ = 2.418032787  # acquisition rate of 2-photon data (Hz)


def load_data(args):
    """Return the raw and warped experimental data."""
    rgb_path = os.path.join(args.experiment, "2Pimg", "RGB.tif")
    wrp_path = os.path.join(args.experiment, "2Pimg", "warped_RGB.tif")
    
    rgb_input = imread_to_float(rgb_path)
    # If warping is not enforced, look for existing warped data
    if not args.force_warp and os.path.isfile(wrp_path):
        wrp_input = imread_to_float(wrp_path)
    else:
        raise NotImplementedError("call to optic flow warping is not enabled yet")
        wrp_input = rgb_input
    
    return rgb_input, wrp_input


# TODO: save potential "cuts" in seg_annot and return them to use in tracking post-processing
def detection(args, wrp_input):
    """Detect ROIs and return the binary segmentation."""
    # Initialiaze PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    model_name = "190510_aug_synth"
    
    # Detect good frames
    indices = similar_frames(wrp_input)
    
    # Make annotations out of them
    annot_projection = wrp_input[indices].mean(0)
    # If not enough frames, process the projection to reduce noise
    if len(indices) < 50:
        annot_projection = filters.gaussian(annot_projection, multichannel=True)
        annot_projection = np.stack([filters.median(to_npint(annot_projection[..., c]))
                                     for c in range(3)], axis=-1) / 255
    seg_annot = segment_projection(annot_projection, min_area=MIN_AREA, separation_border=False)
    # 60%-40% ratio train-validation, with a maximum of 50 frames in total
    n_train = min(int(0.6 * len(indices)), 30)
    n_valid = min(int(0.4 * len(indices)), 20)
    rgb_train = wrp_input[indices[:n_train]]
    rgb_valid = wrp_input[indices[n_train: n_train + n_valid]]
    seg_train = np.array([seg_annot] * len(rgb_train), rgb_train.dtype)
    seg_valid = np.array([seg_annot] * len(rgb_valid), rgb_train.dtype)
    weights_train = compute_weights(seg_train, contour=False, separation=True)
    
    # Load initial model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "..", "data", "models", model_name)
    model = load_model(model_dir, input_channels="RG", u_depth=U_DEPTH, 
                       out1_channels=OUT1_CHANNELS, device=device)
    
    # Fine tunes on annotations
    model_ft = fine_tune(
            model, rgb_train, seg_train, weights_train, rgb_valid, seg_valid,
            data_aug=True, n_iter_min=0, n_iter_max=1000, patience=200,
            batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, verbose=args.verbose)
    
    # Predict the whole experiment
    predictions = predict_stack(
            model_ft, wrp_input, BATCH_SIZE, input_channels="RG",
            transform=lambda stack: normalize_range(pad_transform_stack(stack, U_DEPTH)))
    predictions = torch.sigmoid(predictions)
    segmentations = (predictions > 0.5).numpy().astype(np.bool)
    if MIN_AREA is not None:
        for i in range(len(segmentations)):
            segmentations[i] = morphology.remove_small_objects(segmentations[i], MIN_AREA)
    
    return segmentations, annot_projection, seg_annot


def tracking(args, wrp_input, segmentations, rgb_init, seg_init):
    """Track and return ROI identities."""
    n_updates = 1
    
    # Initialize the model
    identities = np.zeros(segmentations.shape, np.uint8)
    model = InternalModel()
    model.initialize(rgb_init, seg_init)
    
    # Update the model frame by frame
    for n in range(n_updates):
        for i in range(len(segmentations)):
            identities[i] = model.match_frame(wrp_input[i], segmentations[i], time_idx=i)
            model.update(wrp_input[i], identities[i], time_idx=i)
    
    # Track final identities
    for i in range(len(segmentations)):
        identities[i] = model.match_frame(wrp_input[i], segmentations[i], time_idx=i)
    
    # Error detection and correction
    # TODO
    
    # Assure axon ids are consecutive numbers starting at 1
    new_identities = np.zeros_like(identities)
    ids = np.unique(identities)
    ids = np.delete(ids, np.where(ids == 0))
    for new_id, old_id in enumerate(ids):
        new_identities[identities == old_id] = new_id + 1
    
    return new_identities, model


def extract_fluorescence(args, wrp_input, identities):
    """Extract and return the fluorescence traces of each axons as dF/F and dR/R."""
    # Number of frames for baseline computation
    len_baseline = int(BIN_S * RATE_HZ + 0.5)
    
    # Extract fluorophore intensities
    tdtom, gcamp = get_fluorophores(wrp_input, identities)
    
    # Compute fluorescence traces
    dFF, dRR = compute_fluorescence(tdtom, gcamp, len_baseline)
    
    return dFF, dRR


# TODO: maybe saving intermediate results at each step will be enough
def save_results(args, wrp_input, segmentations, identities, dFF, dRR):
    """Save final results."""
    raise NotImplementedError("results saved for testing only")


def main(args):
    """Main function of the AxoID script."""
    # Initialiaze the script
    seed = 1
    random.seed(seed)
    np.random.seed(seed*10 + 1234)
    torch.manual_seed(seed*100 + 4321)
    if args.timeit:
        start = time.time()
    
    # Load and warp the data
    rgb_input, wrp_input = load_data(args)
    
    # Detect ROIs as a binary segmentation
    segmentations, rgb_proj, seg_proj = detection(args, wrp_input)
    
    # Track identities through frames
    identities, model = tracking(args, wrp_input, segmentations, rgb_proj, seg_proj)
    
    # Extract fluorescence traces as both dF/F and dR/R
    dFF, dRR = extract_fluorescence(args, wrp_input, identities)
    
    # Save all results to folder
#    save_results(args, wrp_input, segmentations, rgb_proj, seg_proj, identities, dFF, dRR)
    outdir = "/home/user/talabot/workdir/axoid_outputs/"
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(os.path.join(outdir, "RGB.tif"), to_npint(rgb_input))
        io.imsave(os.path.join(outdir, "warped_RGB.tif"), to_npint(wrp_input))
        io.imsave(os.path.join(outdir, "segmentations.tif"), to_npint(segmentations))
        io.imsave(os.path.join(outdir, "rgb_proj.png"), to_npint(rgb_proj))
        io.imsave(os.path.join(outdir, "seg_proj.png"), to_npint(seg_proj))
        io.imsave(os.path.join(outdir, "identities.tif"), to_npint(identities))
        io.imsave(os.path.join(outdir, "model.png"), to_npint(model.image))
    with open(os.path.join(outdir, "fluorescence.pkl"), "wb") as f:
        pickle.dump({"dFF": dFF, "dRR": dRR}, f)
    
    if args.timeit:
        duration = time.time() - start
        print("AxoID took %d min %d s." % (duration // 60, duration % 60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Main script of AxoID that detects and tracks ROIs on "
            "2-photon neuroimaging data.")
    parser.add_argument(
            'experiment',
            type=str,
            help="path to the experiment folder (excluding \"2Pimg/\")"
    )
    parser.add_argument(
            '--force_warp',
            action="store_true",
            help="force initial optic flow warping of the raw data, even if "
            "\"warped_RGB.tif\" already exists (not necessary if there is no "
            "warped data in the experiment folder)"
    )
    parser.add_argument(
            '--no_gpu', 
            action="store_true",
            help="disable GPU utilization"
    )     
    parser.add_argument(
            '-s', '--seed',
            type=int,
            default=1,
            help="seed for the script (default=1)"
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