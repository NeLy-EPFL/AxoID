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
import matplotlib.pyplot as plt
from scipy import stats
from skimage import filters, io, measure, morphology
import torch

# Add parent folder to path in order to access `axoid`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from axoid.detection.clustering import similar_frames, segment_projection
from axoid.tracking.cutting import find_cuts, apply_cuts
from axoid.detection.deeplearning.model import load_model
from axoid.detection.deeplearning.data import (compute_weights, normalize_range, 
                                               pad_transform_stack)
from axoid.detection.deeplearning.finetuning import fine_tune
from axoid.detection.deeplearning.test import predict_stack
from axoid.tracking.model import InternalModel
from axoid.tracking.utils import renumber_ids
from axoid.utils.image import imread_to_float, to_npint
from axoid.utils.fluorescence import get_fluorophores, compute_fluorescence
from axoid.utils.ccreg import register_stack


## Constants
# Cross-correlation registration
REF_NUM = 1            # Take second frame as reference as first one is often bad
# Deep learning
LEARNING_RATE = 0.0005
BATCH_SIZE = 16
U_DEPTH = 4            # ~depth of U-Net (should be the same as the loaded model)
OUT1_CHANNELS = 16     # ~width of U-Net (should be the same as the loaded model)
MIN_AREA = 11          # minimum size of ROI in pixels
N_ANNOT_MAX = 50       # maximum number of frames to use for fine tuning
TRAIN_RATIO = 0.6      # train-validation ratio for fine tuning
# Tracking
N_UPDATES = 1          # number of pass through whole experiment to update the model
# Fluorescence extraction
BIN_S = 10.0           # bin length for baseline computation (s)
RATE_HZ = 2.418032787  # acquisition rate of 2-photon data (Hz)


def get_data(args):
    """Return the raw and warped experimental data."""
    if args.verbose:
        print("\nGetting input data")
    rgb_path = os.path.join(args.experiment, "2Pimg", "RGB.tif")
    ccreg_path = os.path.join(args.experiment, "2Pimg", "ccreg_RGB.tif")
    wrp_path = os.path.join(args.experiment, "2Pimg", "warped_RGB.tif")
    
    rgb_input = imread_to_float(rgb_path)
    # If warping is not enforced, look for existing warped data
    if not args.force_warp and os.path.isfile(wrp_path):
        if args.verbose:
            print("Warped data found, loading it")
        wrp_input = imread_to_float(wrp_path)
    else:
        if args.verbose:
            print("Starting optic flow warping of input data")
            if args.timeit:
                start = time.time()
        raise NotImplementedError("call to optic flow warping is not enabled yet")
        wrp_input = rgb_input
        if args.verbose and args.timeit:
            duration = time.time() - start
            print("Optic Flow Warping took %d min %d s." % (duration // 60, duration % 60))
    
    # If cc-registration is not enforced, look for existing warped data
    if not args.force_ccreg and os.path.isfile(ccreg_path):
        if args.verbose:
            print("CC registered data found, loading it")
        ccreg_input = imread_to_float(ccreg_path)
    else:
        if args.verbose:
            print("Starting cross-correlation registration of input data")
            if args.timeit:
                start = time.time()
        ccreg_input = register_stack(rgb_input, ref_num=REF_NUM, channels=[0,1])
        if args.verbose and args.timeit:
            print("CC registration took %d s." % (time.time() - start))
    
    return rgb_input, ccreg_input, wrp_input


def _smooth_frame(frame):
    """Return the frame after gaussian smoothing and median filtering."""
    out = filters.gaussian(frame, multichannel=True)
    out = np.stack([filters.median(to_npint(out[..., c]))
                    for c in range(3)], axis=-1) / 255
    return out

def detection(args, input_data, finetuning):
    """Detect ROIs and return the binary segmentation."""
    if args.verbose:
        print("\nDetection of ROIs")
    # Initialiaze PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    if args.verbose:
        print("PyTorch device:", device)
    model_name = "190510_aug_synth"
    
    # Load initial model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "..", "data", "models", model_name)
    model = load_model(model_dir, input_channels="RG", u_depth=U_DEPTH, 
                       out1_channels=OUT1_CHANNELS, device=device)
    
    if finetuning:
        # Detect good frames
        indices = similar_frames(input_data)
        
        # Compute the projection of good frames
        annot_projection = input_data[indices].mean(0)
        # If not enough frames, process the projection to reduce noise
        if len(indices) < N_ANNOT_MAX:
            annot_projection = _smooth_frame(annot_projection)
        seg_annot = segment_projection(annot_projection, min_area=MIN_AREA)
        
        # Make annotations out of the cluster
        # ratio between train&validation, with a maximum number of frames
        n_train = int(TRAIN_RATIO * min(len(indices), N_ANNOT_MAX))
        n_valid = int((1 - TRAIN_RATIO) * min(len(indices), N_ANNOT_MAX))
        rgb_train = input_data[indices[:n_train]]
        rgb_valid = input_data[indices[n_train: n_train + n_valid]]
        seg_train = np.array([seg_annot] * len(rgb_train), rgb_train.dtype)
        seg_valid = np.array([seg_annot] * len(rgb_valid), rgb_train.dtype)
        weights_train = compute_weights(seg_train, contour=False, separation=True)
        
        # Fine tunes on annotations
        if args.verbose:
            print("Fine tuning of network:")
            if args.timeit:
                start = time.time()
        model = fine_tune(
                model, rgb_train, seg_train, weights_train, rgb_valid, seg_valid,
                data_aug=True, n_iter_min=0, n_iter_max=args.maxiter, patience=200,
                batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, verbose=args.verbose)
        if args.verbose and args.timeit:
            duration = time.time() - start
            print("Fine tuning took %d min %d s." % (duration // 60, duration % 60))
    
    # Predict the whole experiment
    if args.verbose and args.timeit and device.type == "cpu": # only time if predicting on cpu
        start = time.time()
    predictions = predict_stack(
            model, input_data, BATCH_SIZE, input_channels="RG",
            transform=lambda stack: normalize_range(pad_transform_stack(stack, U_DEPTH)))
    predictions = torch.sigmoid(predictions)
    
    segmentations = (predictions > 0.5).numpy().astype(np.bool)
    
    if MIN_AREA is not None:
        for i in range(len(segmentations)):
            segmentations[i] = morphology.remove_small_objects(segmentations[i], MIN_AREA)
            
    if args.verbose and args.timeit and device.type == "cpu": # only time if predicting on cpu
        duration = time.time() - start
        print("Prediction of experiment took %d min %d s." % (duration // 60, duration % 60))
    
    if finetuning:
        return segmentations, annot_projection, seg_annot
    else:
        # Find a frame to initialize the tracker model
        # Number of detected ROIs through time
        n_roi = np.zeros(len(input_data), np.uint8)
        for i in range(len(segmentations)):
            _, n = measure.label(segmentations[i], connectivity=1, return_num=True)
            n_roi[i] = n
        
        # Take a frame with #ROIs == mode(#ROIs)
        mode_n_roi = stats.mode(n_roi)[0]
        
        # And with the highest correlation score to the average of the frames with mode(#ROIs) ROIs
        mean_frame = input_data[n_roi == mode_n_roi].mean(0)
        cross_corr = np.sum(input_data[n_roi == mode_n_roi] * mean_frame, axis=(1,2,3))
        
        init_idx = np.argmax(cross_corr)
        # Report the index to the full input stack
        init_idx = np.argmax(np.cumsum(n_roi == mode_n_roi) == init_idx + 1)
        
        # Take the smoothed frame and segment it
        rgb_init = _smooth_frame(input_data[init_idx])
        seg_init = segmentations[init_idx]
        
        return segmentations, rgb_init, seg_init


def tracking(args, input_data, segmentations, rgb_init, seg_init, finetuning):
    """Track and return ROI identities."""   
    if args.verbose:
        print("\nTracking axon identities") 
    
    # Initialize the model
    identities = np.zeros(segmentations.shape, np.uint8)
    model = InternalModel()
    model.initialize(rgb_init, seg_init)
    
    # Compute "cuts" of ROIs
    if finetuning:
        cuts = find_cuts(rgb_init, model, min_area=MIN_AREA)
    
    # Update the model frame by frame
    if args.verbose and args.timeit:
        start = time.time()
    for n in range(N_UPDATES):
        for i in range(len(segmentations)):
            identities[i] = model.match_frame(input_data[i], segmentations[i], time_idx=i)
            model.update(input_data[i], identities[i], time_idx=i)
    if args.verbose and args.timeit:
        duration = time.time() - start
        print("Updating model took %d min %d s." % (duration // 60, duration % 60))
    
    # Track final identities
    if args.verbose and args.timeit:
        start = time.time()
    for i in range(len(segmentations)):
        identities[i] = model.match_frame(input_data[i], segmentations[i], time_idx=i)
    if args.verbose and args.timeit:
        duration = time.time() - start
        print("Identities matching took %d min %d s." % (duration // 60, duration % 60))
    
    # Error detection and correction
    # TODO
    
    # Assure axon ids are consecutive integers starting at 1
    identities = renumber_ids(model, identities)
    
    # Apply "cuts" to model image and all frames
    if finetuning:
        model.image, identities = apply_cuts(cuts, model, identities)
    
    return identities, model


# TODO: maybe saving intermediate results at each step will be enough
def save_results(args, name, i, input_data, segmentations, rgb_proj, seg_proj,
                 identities, model, tdtom, gcamp, dFF, dRR):
    """Save final results."""
    if args.verbose:
        print("Saving results")
    print("Warnings: saving results is currently only implemented for testing!")
    outdir = "/home/user/talabot/workdir/axoid_outputs/"
    if name != "":
        outdir += name + '/'
    if args.animal_folder:
        outdir += str(i) + '/'
    
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(os.path.join(outdir, "input.tif"), to_npint(input_data))
        io.imsave(os.path.join(outdir, "segmentations.tif"), to_npint(segmentations))
        io.imsave(os.path.join(outdir, "rgb_proj.png"), to_npint(rgb_proj))
        io.imsave(os.path.join(outdir, "seg_proj.png"), to_npint(seg_proj))
        io.imsave(os.path.join(outdir, "seg_proj_cut.png"), to_npint(
                segment_projection(rgb_proj, min_area=MIN_AREA, separation_border=True)))
        io.imsave(os.path.join(outdir, "identities.tif"), to_npint(identities))
        io.imsave(os.path.join(outdir, "model.png"), to_npint(model.image))
        
    with open(os.path.join(outdir, "fluorescence.pkl"), "wb") as f:
        pickle.dump({"tdtom": tdtom, "gcamp": gcamp, "dFF": dFF, "dRR": dRR}, f)
    
    def plot_traces(traces, name, color="C2"):
        """Make a plot of the fluorescence traces."""
        ymin = min(0, np.nanmin(traces[:, 1:]) - 0.05 * np.abs(np.nanmin(traces[:, 1:])))
        ymax = np.nanmax(traces[:, 1:]) * 1.05
        fig = plt.figure(figsize=(6, 3 * len(traces)))
        for i in range(len(traces)):
            ax = plt.subplot(len(traces), 1, i+1)
            plt.title("ROI#%d" % i)
            plt.axhline(0, linestyle='dashed', color='gray', linewidth=0.5)
            plt.plot(traces[i], color, linewidth=1)
            plt.xlabel("Frame")
            plt.xlim(0, len(traces[i]) - 1)
            plt.ylabel(name + " (%)")
            plt.ylim(ymin, ymax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i < len(traces) - 1:
                ax.spines['bottom'].set_visible(False)
                ax.get_xaxis().set_visible(False)
        return fig
    fig_tdtom = plot_traces(tdtom, "TdTomato", color="C3")
    fig_gcamp = plot_traces(gcamp, "GCaMP", color="C2")
    fig_dFF = plot_traces(dFF, "$\Delta$F/F")
    fig_dRR = plot_traces(dRR, "$\Delta$R/R")
    fig_tdtom.savefig(os.path.join(outdir, "ROIs_tdTom.png"), bbox_inches='tight')
    fig_gcamp.savefig(os.path.join(outdir, "ROIs_GC.png"), bbox_inches='tight')
    fig_dFF.savefig(os.path.join(outdir, "ROIs_dFF.png"), bbox_inches='tight')
    fig_dRR.savefig(os.path.join(outdir, "ROIs_dRR.png"), bbox_inches='tight')


def process(args, name, i, input_data, finetuning):
    """Apply the AxoID pipeline to the data."""
    if args.verbose:
        print("\n\t # Processing %s #" % name)
        if args.timeit:
            start = time.time()
    
    # Detect ROIs as a binary segmentation
    segmentations, rgb_proj, seg_proj = detection(args, input_data, finetuning)
    
    # Track identities through frames
    identities, model = tracking(args, input_data, segmentations,
                                 rgb_proj, seg_proj, finetuning)
    
    # Extract fluorescence traces as both dF/F and dR/R
    if args.verbose:
        print("\nExtracting fluorescence traces")
        if args.timeit:
            substart = time.time()
    len_baseline = int(BIN_S * RATE_HZ + 0.5) # number of frames for baseline computation
    tdtom, gcamp = get_fluorophores(input_data, identities)
    dFF, dRR = compute_fluorescence(tdtom, gcamp, len_baseline)
    if args.verbose and args.timeit:
        print("Fluorescence extraction took %d s." % (time.time() - substart))
    
    # Save all results to folder
    save_results(args, name, i, input_data, segmentations, rgb_proj, seg_proj,
                 identities, model, tdtom, gcamp, dFF, dRR)
    
    if args.verbose and args.timeit:
        duration = time.time() - start
        print("\nProcessing %s took %d min %d s." % 
              (name, duration // 60, duration % 60))


def main(args):
    """Main function of the AxoID script."""
    # Initialiaze the script
    random.seed(args.seed)
    np.random.seed(args.seed*10 + 1234)
    torch.manual_seed(args.seed*100 + 4321)
    if args.timeit:
        start = time.time()
    if args.verbose:
        print("AxoID started on " + time.asctime())
    
    # Make a list of the experiment folders to process
    if args.animal_folder:
        experiments = [os.path.join(args.experiment, folder) 
                       for folder in os.listdir(args.experiment)
                       if os.path.isdir(os.path.join(args.experiment, folder))]
    else:
        experiments = [args.experiment]
    
    for i, exp in enumerate(experiments):
        args.experiment = exp
        if args.verbose and len(experiments) > 1:
            print("\nProcessing experiment %d/%d: (%s)" % 
                  (i + 1, len(experiments), exp))
            if args.timeit:
                start_exp = time.time()
        
        # Load and warp the data
        rgb_input, ccreg_input, wrp_input = get_data(args)
        
        # Apply the full AxoID pipeline to the inputs
        process(args, "Raw", i, rgb_input, finetuning=False)
        process(args, "CCReg", i, ccreg_input, finetuning=True)
        process(args, "OFW", i, wrp_input, finetuning=True)
        
        if args.verbose and args.timeit and len(experiments) > 1:
            duration = time.time() - start_exp
            print("\Processing the experiment took %d min %d s." % 
                  (duration // 60, duration % 60))
    
    if args.timeit:
        duration = time.time() - start
        print("\nAxoID took %d min %d s." % (duration // 60, duration % 60))
    if args.verbose:
        print("AxoID finished on " + time.asctime())


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
            '--animal_folder',
            action="store_true",
            help="if this is used, the passed folder name is assumed to be an "
            "animal folder regrouping multiple experiment folders in it, AxoID "
            "will then be applied sequentially to each"
    )
    parser.add_argument(
            '--force_ccreg',
            action="store_true",
            help="force initial cross-correlation registration of the raw data,"
            " even if \"ccreg_RGB.tif\" already exists"
    )
    parser.add_argument(
            '--force_warp',
            action="store_true",
            help="force initial optic flow warping of the raw data, even if "
            "\"warped_RGB.tif\" already exists"
    )
    parser.add_argument(
            '--maxiter',
            type=int,
            default=1000,
            help="maximum number of iteration for the network fine tuning "
            "(default=1000)"
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
            help="time the script. If --verbose is used, sub-parts of the script "
            "will also be timed."
    )
    parser.add_argument(
            '-v', '--verbose', 
            action="store_true",
            help="enable output verbosity"
    )
    args = parser.parse_args()

    main(args)