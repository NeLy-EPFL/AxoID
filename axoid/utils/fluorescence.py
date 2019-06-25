#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions for extracting fluorescence traces.
Created on Thu Jun  6 14:49:55 2019

@author: nicolas
"""

import os.path
import warnings
import pickle

import numpy as np
import matplotlib.pyplot as plt


def get_fluorophores(rgb_stack, identities):
    """
    Return the tdTomato and GCaMP time series of each ROI.
    
    Parameters
    ----------
    rgb_stack : ndarray
        Stack of RGB images with the tdTomato fluorophore in the red channel,
        and GCaMP fluorophore in the green channel (blue is ignored).
    identities : ndarray of int
        Stack of ROI images with their identity as value, 0 is background.
    
    Returns
    -------
    tdtom : ndarray
        Time series of average tdTomato intensity for each axon. If an axon is 
        not present on a frame, its intensity is set to np.nan for that frame.
        Shape is n_axons x n_images.
    gcamp : ndarray
        Same, but for the GCaMP fluorohpore.
    """
    # Find unique id of each axon (remove 0 which is background)
    ids = np.unique(identities)
    ids = np.delete(ids, np.where(ids == 0))
    
    tdtom = np.zeros((len(ids), len(rgb_stack)))
    gcamp = np.zeros((len(ids), len(rgb_stack)))
    
    # Loop over axons
    for n, id in enumerate(ids):
        # Mask of ROI corresponding to axon
        roi_mask = identities == id
        
        # Mask per channel with only non-zero pixels taken into account
        tdtom_mask = np.logical_and(roi_mask, rgb_stack[..., 0] > 0)
        gcamp_mask = np.logical_and(roi_mask, rgb_stack[..., 1] > 0)
        
        # Compute the average intensity of unmasked pixels
        # NaN in frame where an axon is absent (so we discard warnings for this)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tdtom[n] = (rgb_stack[..., 0] * tdtom_mask).sum((1, 2)) / tdtom_mask.sum((1, 2))
            gcamp[n] = (rgb_stack[..., 1] * gcamp_mask).sum((1, 2)) / gcamp_mask.sum((1, 2))
    
    return tdtom, gcamp

def compute_fluorescence(tdtom, gcamp, len_baseline):
    """
    Compute and return dF/F and dR/R fluorescence traces.
    
    Parameters
    ----------
    tdtom : ndarray
        Time series of average tdTomato intensity for each axon. If an axon is 
        not present on a frame, its intensity is set to np.nan for that frame.
        Shape is n_axons x n_images.
    gcamp : ndarray
        Same, but for the GCaMP fluorohpore.
    len_baseline : int
        Number of consecutive frames to consider for the baseline computation.
    
    Returns
    -------
    dFF : ndarray
        Fluorescence traces for each axon computed as dF/F, in percent.
        Shape is n_axons x n_images.
    dRR : ndarray
        Same, but computed as dR/R.
    """
    dFF = np.zeros_like(gcamp)
    dRR = np.zeros_like(gcamp)
    
    # Loop on axons
    for n in range(dFF.shape[0]):
        F_t = gcamp[n].copy()
        R_t = gcamp[n] / tdtom[n]
        
        # Compute baselines by looping on frames
        F_0 = []
        R_0 = []
        for i in range(max(0, dFF.shape[1] - len_baseline)):
            # Ignore empty slice warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                F_0.append(np.nanmean(F_t[i: i + len_baseline]))
                R_0.append(np.nanmean(R_t[i: i + len_baseline]))
        F_0 = np.nanmin(F_0)
        R_0 = np.nanmin(R_0)
        
        dFF[n] = (F_t - F_0) / F_0 * 100
        dRR[n] = (R_t - R_0) / R_0 * 100
    
    return dFF, dRR

def save_fluorescence(path, tdtom, gcamp, dFF, dRR):
    """Save the fluorescence traces in the experiment folder."""
    # Save traces
    def save_traces(traces, filename):
        """Pickle and save the traces under filename.pkl."""
        data_dic = {"ROI_" + str(i): traces[i] for i in range(len(traces))}
        with open(os.path.join(path, filename + ".p"), "wb") as f:
            pickle.dump(data_dic, f)
    save_traces(tdtom, "tdTom_abs_dic")
    save_traces(gcamp, "GC_abs_dic")
    save_traces(dFF, "dFF_dic")
    save_traces(dRR, "dRR_dic")        
    # Save plots of the traces
    def plot_traces(traces, filename, ylabel, ylim=None):
        """Make a plot of the fluorescence traces."""
        color="forestgreen"
        if ylim is not None:
            ymin, ymax = ylim
        else:
            ymin = min(0, np.nanmin(traces[:, 1:]) - 0.05 * np.abs(np.nanmin(traces[:, 1:])))
            ymax = np.nanmax(traces[:, 1:]) * 1.05
        
        fig = plt.figure(figsize=(8, 4 * len(traces)), facecolor='white', dpi=300)
        fig.subplots_adjust(left=0.2, right = 0.9, wspace = 0.3, hspace = 0.3)
        
        for i in range(len(traces)):
            ax = plt.subplot(len(traces), 1, i+1)
            plt.axhline(linestyle='dashed', color='gray', linewidth=0.5)
            plt.plot(traces[i], color, linewidth=1)
            plt.xlim(0, 1.05*(len(traces[i]) - 1))
            plt.ylabel("ROI#%d\n" % i + ylabel, size=10, color=color)
            plt.ylim(ymin, ymax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i < len(traces) - 1:
                ax.spines['bottom'].set_visible(False)
                ax.get_xaxis().set_visible(False)
            else:
                plt.xlabel("Frame", size=10)
                
        fig.savefig(os.path.join(path, filename + ".png"),
                    bbox_inches='tight', facecolor=fig.get_facecolor(),
                    edgecolor='none', transparent=True)
    
    plot_traces(tdtom, "ROIS_tdTom", ylabel="F", ylim=(-0.05, 1.05))
    plot_traces(gcamp, "ROIS_GC", ylabel="F", ylim=(-0.05, 1.05))
    plot_traces(dFF, "ROIs_dFF", ylabel="$\Delta$F/F (%)")
    plot_traces(dRR, "ROIs_dRR", ylabel="$\Delta$R/R (%)")