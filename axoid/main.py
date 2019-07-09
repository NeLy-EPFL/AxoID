#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main functions of AxoID with the full pipeline.
It runs AxoID on experiment, detecting and tracking ROIs, then
extracting fluorescence traces and saving outputs.
Created on Wed Jun 26 13:30:52 2019

@author: nicolas
"""

import os, os.path, sys
import subprocess
import warnings
import shutil
import pickle
import time
import argparse
import random

import numpy as np
from scipy import stats
from skimage import filters, io, measure, morphology
import cv2
import torch

from .detection.clustering import similar_frames, segment_projection
from .detection.deeplearning.model import load_model
from .detection.deeplearning.data import (compute_weights, normalize_range, 
                                          pad_transform_stack)
from .detection.deeplearning.finetuning import fine_tune
from .detection.deeplearning.test import predict_stack
from .tracking.model import InternalModel
from .tracking.utils import renumber_ids
from .tracking.cutting import find_cuts, apply_cuts
from .utils.image import imread_to_float, to_npint, gray2red
from .utils.fluorescence import get_fluorophores, compute_fluorescence, save_fluorescence
from .utils.ccreg import register_stack, shift_image
# Add parent folder to path in order to access `motion_compensation_path`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Command for optic flow warping through command line
from motion_compensation_path import COMMAND as OFW_COMMAND


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
    """
    Return the experimental data.
    
    Parameters
    ----------
    args : arguments
        Arguments passed to the script through the command line.
    
    Returns
    -------
    rgb_input : ndarray (float)
        Stack of images with the raw experimental data.
    ccreg_input : ndarray (float)
        Stack of images with the cross-correlation registered experimental data.
    wrp_input : ndarray (float)
        Stack of images with the optic flow warped experimental data,
        for detection and tracking.
    wrp_fluo : ndarray (float)
        Stack of images with the optic flow warped experimental data,
        for fluorescence extraction.
    """
    if args.verbose:
        print("\nGetting input data")
    rgb_path = os.path.join(args.experiment, "2Pimg", "RGB.tif")
    wrp_path = os.path.join(args.experiment, "2Pimg", "warped_RGB.tif")
    
    ## raw data
    rgb_input = imread_to_float(rgb_path)
    rgb_fluo = np.stack([
            imread_to_float(os.path.join(args.experiment, "2Pimg", "tdTom.tif")),
            imread_to_float(os.path.join(args.experiment, "2Pimg", "GC6fopt.tif")),
            imread_to_float(os.path.join(args.experiment, "2Pimg", "GC6fopt.tif"))
    ], axis=-1)
    
    ## Always cross-correlation registered the data
    if args.verbose:
        print("Starting cross-correlation registration of input data")
        if args.timeit:
            start = time.time()
            
    ccreg_input, rows, cols = register_stack(rgb_input, ref_num=REF_NUM,
                                             channels=[0,1], return_shifts=True)
    # Register the fluo data
    ccreg_fluo = rgb_fluo.copy()
    for i in range(len(ccreg_fluo)):
        ccreg_fluo[i] = shift_image(rgb_fluo[i], rows[i], cols[i])
    
    if args.verbose and args.timeit:
        print("CC registration took %d s." % (time.time() - start))
    
    ## If warping is not enforced, look for existing warped data
    if not args.force_warp and os.path.isfile(wrp_path):
        if args.verbose:
            print("warped_RGB.tif found, loading it")
        wrp_input = imread_to_float(wrp_path)
    else:
        wrp_input = None
    # Look for warped data to use for fluorescence computation
    # This is different from warped_RGB.tif which has been manually edited
    if not args.force_warp:
        if args.warpdir is not None:
            folder = args.warpdir
        elif os.path.isdir(os.path.join(args.experiment, "2Pimg", "results_GreenRed")):
            folder = "results_GreenRed"
        elif os.path.isdir(os.path.join(args.experiment, "2Pimg", "results")):
            folder = "results"
        elif wrp_input is not None:
            folder = None
            if args.verbose:
                print("Warped data for fluorescence not found, copying warped_RGB.tif")
            wrp_fluo = wrp_input.copy()
        else:
            folder = None
            wrp_fluo = None
            print("No warped data was found")
        
        if folder is not None:
            folder = os.path.join(args.experiment, "2Pimg", folder)
            folder = os.path.join(folder, os.listdir(folder)[0])
            wrp_tdtom = imread_to_float(os.path.join(folder, "warped1.tif"))
            wrp_gcamp = imread_to_float(os.path.join(folder, "warped2.tif"))
            wrp_fluo = np.stack([wrp_tdtom, wrp_gcamp, wrp_gcamp], axis=-1)
    # If forcing warping or no data is available, apply warping
    if args.force_warp or wrp_fluo is None:
        if args.verbose:
            print("Starting optic flow warping of input data")
            if args.timeit:
                start = time.time()
                
        # Call motion_compensation MATLAB script to warp the data
        # Make temp folder, save tdTom and GCaMP w\ 1st frame
        tmpdir = os.path.join(args.experiment, "output", "axoid_internal", "tmp_ofw")
        os.makedirs(tmpdir, exist_ok=True)
        io.imsave(os.path.join(tmpdir, "red.tif"), to_npint(rgb_input[1:,...,0]))
        io.imsave(os.path.join(tmpdir, "green.tif"), to_npint(rgb_input[1:,...,1]))
        
        # Give it to OFW through a subprocess
        results_dir = os.path.join(args.experiment, "output", "axoid_internal",
                                   "warped_results")
        wrp_process = subprocess.run(
                OFW_COMMAND.split(" ") + [tmpdir, "-l", str(args.warp_l),
                "-g", str(args.warp_g), "-results_dir", results_dir],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False, universal_newlines=True)
        
        # Save warping logs regardless of exit status
        log_path = os.path.join(args.experiment, "output", "axoid_internal",
                                "logs_warping.txt")
        with open(log_path, "w") as f:
            f.write(wrp_process.stdout)
        if args.verbose:
            print("Warping logs saved at %s." % log_path)
        
        # Then, check wether the subprocess failed
        wrp_process.check_returncode()
        
        # Load outputs of warping, and make a `wrp_input stack`
        wrp_tdtom = imread_to_float(os.path.join(results_dir, 
            "l%dg%d" % (args.warp_l, args.warp_g), "red_warped.tif"))
        wrp_gcamp = imread_to_float(os.path.join(results_dir, 
            "l%dg%d" % (args.warp_l, args.warp_g), "green_warped.tif"))
        wrp_input = np.stack([wrp_tdtom, wrp_gcamp, wrp_gcamp], axis=-1)
        
        # Re-add the (unwarped) first frame and remove temp folder
        wrp_tdtom = np.concatenate([rgb_input[np.newaxis,0,...,0], wrp_tdtom], axis=0)
        wrp_gcamp = np.concatenate([rgb_input[np.newaxis,0,...,1], wrp_gcamp], axis=0)
        wrp_input = np.concatenate([rgb_input[np.newaxis,0], wrp_input], axis=0)
        wrp_fluo = wrp_input.copy()
        shutil.rmtree(tmpdir)
        
        if args.verbose and args.timeit:
            duration = time.time() - start
            print("Optic Flow Warping took %d min %d s." % (duration // 60, duration % 60))
            
        # Save the warped stack and temporal projection
        wrpdir = os.path.join(args.experiment, "2Pimg", "results", 
                              "l%dg%d" % (args.warp_l, args.warp_g))
        os.makedirs(wrpdir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Warning)
            io.imsave(os.path.join(wrpdir, "warped1.tif"), 
                      to_npint(wrp_tdtom))
            io.imsave(os.path.join(wrpdir, "warped2.tif"), 
                      to_npint(wrp_gcamp))
    
    if wrp_input is None:
        if args.verbose:
            print("warped_RGB.tif not found, copying warped data from results folder")
        wrp_input = wrp_fluo.copy()
    
    # Save inputs to internal folder
    outdir = os.path.join(args.experiment, "output", "axoid_internal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        io.imsave(os.path.join(outdir, "raw", "input.tif"), to_npint(rgb_input))
        io.imsave(os.path.join(outdir, "raw", "input_fluo.tif"), to_npint(rgb_fluo))
        io.imsave(os.path.join(outdir, "ccreg", "input.tif"), to_npint(ccreg_input))
        io.imsave(os.path.join(outdir, "ccreg", "input_fluo.tif"), to_npint(ccreg_fluo))
        io.imsave(os.path.join(outdir, "warped", "input.tif"), to_npint(wrp_input))
        io.imsave(os.path.join(outdir, "warped", "input_fluo.tif"), to_npint(wrp_fluo))
    
    return rgb_input, rgb_fluo, ccreg_input, ccreg_fluo, wrp_input, wrp_fluo


def _smooth_frame(frame):
    """Return the RGB frame after gaussian smoothing and median filtering."""
    out = filters.gaussian(frame, multichannel=True)
    out = np.stack([filters.median(to_npint(out[..., c]))
                    for c in range(3)], axis=-1) / 255
    return out

def detection(args, name, input_data, finetuning):
    """
    Detect ROIs and return the binary segmentation.
    
    Parameters
    ----------
    args : arguments
        Arguments passed to the script through the command line.
    name : str
        Name of the current folder/type of input to process: "raw", "ccreg", 
        or "warped".
    input_data : ndarray
        Image stack of the input on which to detect ROIs.
    finetuning : bool
        If true, will look for similar frames and finetune on them. If False,
        no finetuning is applied.
    
    Returns
    -------
    segmentations : ndarray (bool)
        Stack of binary images representing the ROI detection.
    rgb_init : ndarray (float)
        Frame to use to initialize the tracker.
    seg_init : ndarray (float)
        Binary detection of `rgb_init` to use to initialize the tracker.
    """
    if args.verbose:
        print("\nDetection of ROIs")
    # Initialiaze PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    if args.verbose:
        print("PyTorch device:", device)
    
    outdir = os.path.join(args.experiment, "output", "axoid_internal", name)
    
    # Load initial model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "..", "model")
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
        
        # Save intermediate results
        with open(os.path.join(outdir, "indices_init.txt"), "w") as f:
            for idx in indices:
                f.write(str(idx) + "\n")
        seg_annot_cut = segment_projection(annot_projection, min_area=MIN_AREA, 
                                           separation_border=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Warning)
            io.imsave(os.path.join(outdir, "rgb_init.tif"), 
                      to_npint(annot_projection))
            io.imsave(os.path.join(outdir, "seg_init.tif"), 
                      to_npint(seg_annot))
            io.imsave(os.path.join(outdir, "seg_init_cut.tif"), 
                      to_npint(seg_annot_cut))
        
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
       
    # Save segmentations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        io.imsave(os.path.join(outdir, "segmentations.tif"), 
                  to_npint(segmentations))
    
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
        
        # And the highest correlation score to the average of the frames with mode(#ROIs) ROIs
        mean_frame = input_data[n_roi == mode_n_roi].mean(0)
        cross_corr = np.sum(input_data[n_roi == mode_n_roi] * mean_frame, 
                            axis=(1,2,3))
        
        init_idx = np.argmax(cross_corr)
        # Report the index to the full input stack
        init_idx = np.argmax(np.cumsum(n_roi == mode_n_roi) == init_idx + 1)
        
        # Take the smoothed frame and segment it
        rgb_init = _smooth_frame(input_data[init_idx])
        seg_init = segmentations[init_idx]
        
        # Save initialization frame
        with open(os.path.join(outdir, "indices_init.txt"), "w") as f:
            f.write(str(init_idx))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Warning)
            io.imsave(os.path.join(outdir, "rgb_init.tif"), to_npint(rgb_init))
            io.imsave(os.path.join(outdir, "seg_init.tif"), to_npint(seg_init))
        
        return segmentations, rgb_init, seg_init


def tracking(args, name, input_data, segmentations, rgb_init, seg_init, finetuning):
    """
    Track and return ROI identities.
    
    Parameters
    ----------
    args : arguments
        Arguments passed to the script through the command line.
    name : str
        Name of the current folder/type of input to process: "raw", "ccreg", 
        or "warped".
    input_data : ndarray
        Image stack of the input on which to detect ROIs.
    segmentations : ndarray
        Stack of binary images of ROI detections.
    rgb_init : ndarray
        Frame to use to initialize the tracker model.
    seg_init : ndarray
        Segmentation of the frame to use to initialize the tracker model.
    finetuning : bool
        If true, will look for similar frames and finetune on them. If False,
        no finetuning is applied.
    
    Returns
    -------
    identities : ndarray (uint8)
        Stack of greyscale images where the grey level corresponds to the
        identity of the pixel (0 is background, 1 and above are IDs).
    model : internal model
        Model of the tracker.
    """   
    if args.verbose:
        print("\nTracking axon identities") 
    outdir = os.path.join(args.experiment, "output", "axoid_internal", name)
    
    # Initialize the model
    identities = np.zeros(segmentations.shape, np.uint8)
    model = InternalModel()
    model.initialize(rgb_init, seg_init)
    
    # Compute "cuts" of ROIs
    if finetuning:
        cuts = find_cuts(rgb_init, model, min_area=MIN_AREA)
        with open(os.path.join(outdir, "cuts.pkl"), "wb") as f:
            pickle.dump(cuts, f)
    
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
    
    # Assure axon ids are consecutive integers starting at 1
    identities = renumber_ids(model, identities)
    
    # Apply "cuts" to model image and all frames
    if finetuning:
        # Save results before cuts are applid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Warning)
            io.imsave(os.path.join(outdir, "identities_precut.tif"), 
                      to_npint(identities))
            io.imsave(os.path.join(outdir, "model_precut.tif"), 
                      to_npint(model.image))
        
        model.image, identities = apply_cuts(cuts, model, identities)
    
    # Save resulting identities and model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        io.imsave(os.path.join(outdir, "identities.tif"), to_npint(identities))
        io.imsave(os.path.join(outdir, "model.tif"), to_npint(model.image))
    
    return identities, model


def save_results(args, name, input_data, identities, tdtom, gcamp, dFF, dRR):
    """
    Save final results.
    
    Parameters
    ----------
    args : arguments
        Arguments passed to the script through the command line.
    name : str
        Name of the current folder/type of input to process: "raw", "ccreg", 
        or "warped".
    input_data : ndarray
        Image stack of the input on which to detect ROIs.
    identities : ndarray
        Stack of identity images.
    tdtom : ndarray
        Time series of tdTom fluorescence values for each axon.
    gcamp : ndarray
        Time series of GCaMP fluorescence values for each axon.
    dFF : ndarray
        Time series of fluorescence values for each axon, computed as dF/F.
    dRR : ndarray
        Time series of fluorescence values for each axon, computed as dR/R.
    """
    if args.verbose:
        print("Saving results")
    
    # Save input + contours stack and contour list
    ids = np.unique(identities)
    ids = ids[ids != 0]
    input_roi = to_npint(input_data)
    contour_list = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        for i in range(len(input_roi)):
            contour_list.append([])
            for n, id in enumerate(ids):
                roi_img = (identities[i] == id)
                if np.sum(roi_img) == 0:
                    contour_list[i].append([])
                    continue
                _, contours, _ = cv2.findContours(roi_img.astype(np.uint8),
                                                  cv2.RETR_LIST,
                                                  cv2.CHAIN_APPROX_NONE)
                x, y = contours[0][:,0,0].max(), contours[0][:,0,1].min() # top-right corner of bbox
                # Draw the contour and write the ROI id
                cv2.drawContours(input_roi[i], contours, -1, (255,255,255), 1)
                cv2.putText(input_roi[i], str(n), (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
                contour_list[i].append(contours[0])
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        io.imsave(os.path.join(args.experiment, "output", "ROI_auto", name,
                               "RGB_seg.tif"), to_npint(input_roi))
    with open(os.path.join(args.experiment, "output", "GC6_auto", name, 
                           "All_contour_list_dic_Usr.p"), "wb") as f:
        pickle.dump({"AllContours": contour_list}, f)
    
    # Save traces
    save_fluorescence(os.path.join(args.experiment, "output", "GC6_auto", name),
                      tdtom, gcamp, dFF, dRR)


def process(args, name, input_data, fluo_data=None, finetuning=True):
    """
    Apply the AxoID pipeline to the data.
    
    
    Parameters
    ----------
    args : arguments
        Arguments passed to the script through the command line.
    name : str
        Name of the current folder/type of input to process: "raw", "ccreg", 
        or "warped".
    input_data : ndarray
        Image stack of the input on which to detect ROIs.
    fluo_data : ndarray
        Image stack of the experimental data to use for fluorescence extraction.
    finetuning : bool
        If true, will look for similar frames and finetune on them. If False,
        no finetuning is applied.
    """
    if args.verbose:
        print("\n\t # Processing %s #" % name)
        if args.timeit:
            start = time.time()
    
    # Detect ROIs as a binary segmentation
    segmentations, rgb_proj, seg_proj = detection(args, name, input_data, finetuning)
    
    # Track identities through frames
    identities, model = tracking(args, name, input_data, segmentations,
                                 rgb_proj, seg_proj, finetuning)
    
    # Extract fluorescence traces as both dF/F and dR/R
    if args.verbose:
        print("\nExtracting fluorescence traces")
        if args.timeit:
            substart = time.time()
            
    len_baseline = int(BIN_S * RATE_HZ + 0.5) # number of frames for baseline computation
    if fluo_data is None:
        tdtom, gcamp = get_fluorophores(input_data, identities)
    else:
        tdtom, gcamp = get_fluorophores(fluo_data, identities)
    dFF, dRR = compute_fluorescence(tdtom, gcamp, len_baseline)
    
    if args.verbose and args.timeit:
        print("Fluorescence extraction took %d s." % (time.time() - substart))
    
    # Save results to folder
    save_results(args, name, input_data, identities, tdtom, gcamp, dFF, dRR)
    
    if args.verbose and args.timeit:
        duration = time.time() - start
        print("\nProcessing %s took %d min %d s." % 
              (name, duration // 60, duration % 60))


def main(args=None):
    """Main function of the AxoID script."""
    if args is None:
        args = parser()
    
    # Initialiaze the script
    random.seed(args.seed)
    np.random.seed(args.seed*10 + 1234)
    torch.manual_seed(args.seed*100 + 4321)
    if args.timeit:
        start = time.time()
    if args.verbose:
        print("AxoID started on " + time.asctime())
    
    # Create output directory
    os.makedirs(os.path.join(args.experiment, "output", "axoid_internal"), exist_ok=True)
    # Make folders and subfolders
    for name in ["raw", "ccreg", "warped"]:
        os.makedirs(os.path.join(args.experiment, "output", "axoid_internal", name), exist_ok=True)
        os.makedirs(os.path.join(args.experiment, "output", "GC6_auto", name), exist_ok=True)
        os.makedirs(os.path.join(args.experiment, "output", "ROI_auto", name), exist_ok=True)
    
    # Load and register the data
    rgb_input, rgb_fluo, ccreg_input, ccreg_fluo, wrp_input, wrp_fluo = get_data(args)
    
    # Apply the full AxoID pipeline to the inputs
    #  Currently the calls are sequential. It should be possible to 
    #  parallelize them, except for the detection part as they would all 
    #  require access to the GPU (which could cause out of memory errors)
    process(args, "raw", rgb_input, rgb_fluo, finetuning=False)
    process(args, "ccreg", ccreg_input, ccreg_fluo, finetuning=True)
    process(args, "warped", wrp_input, wrp_fluo, finetuning=True)
    
    if args.timeit:
        duration = time.time() - start
        print("\nAxoID took %d min %d s." % (duration // 60, duration % 60))
    if args.verbose:
        print("AxoID finished on " + time.asctime())


def parser():
    """
    Parse the command for arguments.
    
    Returns
    -------
    args : arguments
        Arguments passed to the script through the command line.
    """
    parser = argparse.ArgumentParser(
            description="Main script of AxoID that detects and tracks ROIs on "
            "2-photon neuroimaging data.")
    parser.add_argument(
            'experiment',
            type=str,
            help="path to the experiment folder (excluding \"2Pimg/\")"
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
    parser.add_argument(
            '--warpdir',
            type=str,
            default=None,
            help="directory where the warped output is stored inside 2Pimg/"
    )
    parser.add_argument(
            '--warp_g',
            type=int,
            default=10,
            help="gamma parameter for the optic flow warping (default=10)"
    )
    parser.add_argument(
            '--warp_l',
            type=int,
            default=300,
            help="lambda parameter for the optic flow warping (default=300)"
    )
    args = parser.parse_args()

    return args
