#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions/classes for data manipulation with PyTorch.
Created on Mon Oct 22 13:54:19 2018

@author: nicolas
"""

import os
import numpy as np

import torch
from torch.utils import data

from utils_common.image import imread_to_float


class ImageLoaderDataset(data.Dataset):
    """Dataset that loads image online for efficient memory usage."""
    
    def __init__(self, x_filenames, y_filenames, w_filenames=None, 
                 input_channels="RGB", transform=None, target_transform=None):
        """
        Args:
            x_filenames: list of str
                Contains the filenames/path to the input images.
            y_filenames: list of str
                Contains the filenames/path to the target images.
            w_filenames: list of str
                Contains the filenames/path to the weight images.
            input_channels: str (default = "RGB")
                Indicates the channels to load from the input images, e.g. "RG"
                for Red and Green.
            transform: callable (default = None)
                Transformation to apply to the input images.
            target_transform: callable (default = None)
                Transformation to apply to the target and mask images.
        """
        super(ImageLoaderDataset, self).__init__()
        self.x_filenames = x_filenames
        self.y_filenames = y_filenames
        self.w_filenames = w_filenames
        
        if len(self.x_filenames) != len(self.y_filenames):
            raise ValueError("Not the same number of files in input and target lists"
                             " (%d != %d)." % (len(self.x_filenames), len(self.y_filenames)))
            
        self.input_channels = input_channels
        if self.input_channels == "":
            raise ValueError("At least one input channel is needed.")
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.x_filenames)
    
    def __getitem__(self, idx):
        # Load images to float in range [0,1]
        image = imread_to_float(self.x_filenames[idx], scaling=255)
        target = imread_to_float(self.y_filenames[idx], scaling=255)
        if self.w_filenames is not None:
            if self.w_filenames[idx] is not None:
                weight = imread_to_float(self.w_filenames[idx], scaling=255)
            else:
                weight = np.zeros_like(target)
        
        # Keep only relevant input channels
        channel_imgs = {"R": image[:,:,0], "G": image[:,:,1], "B": image[:,:,2]}
        image = np.stack([channel_imgs[channel] for channel in self.input_channels], axis=0)
        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
            if self.w_filenames is not None:
                weight = self.target_transform(weight)
        
        if self.w_filenames is not None:
            return image, target, weight
        else:
            return image, target


def get_filenames(data_dir, use_weights=False, valid_extensions=('.png', '.jpg', '.jpeg')): 
    """
    Return 2(3) lists with the input, target (and weight) filenames respectively.
    Note thate filenames should be order the same to identify correct tuples.
    
    The data directory is assumed to be organised as follow:
        data_dir:
            subdir1:
                rgb_frames: folder with input images
                seg_frames: folder with target images
                wgt_frames: folder with weight images (optional)
            subdir2:
                rgb_frames: folder with input images
                seg_frames: folder with target images
                wgt_frames: folder with weight images (optional)
            ...
    data_dir can also be a list of the path to subdirs to use.
    
    Args:
        data_dir: str, or list of str
            Directory/path to the data, or list of directories/paths to the subdirs.
        use_weights: bool (default = False)
            If True, will look for weight images and add them to the dataloaders.
        valid_extensions: tuple of str (default = ('.png', '.jpg', '.jpeg'))
            Tuple of the valid image extensions.
    
    Returns:
        x_filenames, y_filenames(, w_filenames): lists of str 
            Contain the input, target, (and weight) image paths respectively.
    """
    if isinstance(data_dir, list):
        subdirs_list = data_dir
    else:
        subdirs_list = [os.path.join(data_dir, subdir) for subdir in sorted(os.listdir(data_dir))]
    
    if not isinstance(valid_extensions, tuple):
        valid_extensions = tuple(valid_extensions)
    x_filenames = []
    y_filenames = []
    if use_weights:
        w_filenames = []
    
    for data_subdir in subdirs_list:
        # Inputs
        for frame_filename in sorted(os.listdir(os.path.join(data_subdir, "rgb_frames"))):
            if frame_filename.lower().endswith(valid_extensions):
                x_filenames.append(os.path.join(data_subdir, "rgb_frames", frame_filename))
        # Targets
        for frame_filename in sorted(os.listdir(os.path.join(data_subdir, "seg_frames"))):
            if frame_filename.lower().endswith(valid_extensions):
                y_filenames.append(os.path.join(data_subdir, "seg_frames", frame_filename))
        # Weights
        if use_weights:
            if os.path.isdir(os.path.join(data_subdir, "wgt_frames")):
                for frame_filename in sorted(os.listdir(os.path.join(data_subdir, "wgt_frames"))):
                    if frame_filename.lower().endswith(valid_extensions):
                        w_filenames.append(os.path.join(data_subdir, "wgt_frames", frame_filename))
            else: # if no weights, append None
                for _ in range(len(x_filenames) - len(w_filenames)):
                    w_filenames.append(None)
    
    if use_weights:
        return x_filenames, y_filenames, w_filenames
    else:
        return x_filenames, y_filenames


def _pad_collate(batch):
    """Collate function that pads input/target/mask images to the same size."""
    pad_batch = []
    
    # Find largest shape (item[1] is the target image)
    shapes = [item[1].shape for item in batch]
    heights = np.array([height for height, width in shapes])
    widths = np.array([width for height, width in shapes])
    max_height = np.max(heights)
    max_width = np.max(widths)
    # If all of the same size, don't pad
    if (heights == max_height).all() and (widths == max_width).all():
        return data.dataloader.default_collate(batch)
    
    # Pad images to largest shape 
    for item in batch:
        shape = item[1].shape
        padding = [(int(np.floor((max_height - shape[0])/2)), int(np.ceil((max_height - shape[0])/2))), 
                   (int(np.floor((max_width - shape[1])/2)), int(np.ceil((max_width - shape[1])/2)))]
        if len(item) == 2: # no weight
            pad_batch.append((
                np.pad(item[0], [(0,0)] + padding, 'constant'),
                np.pad(item[1], padding, 'constant')))
        else:
            pad_batch.append((
                np.pad(item[0], [(0,0)] + padding, 'constant'),
                np.pad(item[1], padding, 'constant'),
                np.pad(item[2], padding, 'constant')))
    
    return data.dataloader.default_collate(pad_batch)


def get_dataloader(data_dir, batch_size, input_channels="RG",
                   shuffle=True,  use_weights=False,
                   transform=None, target_transform=None, num_workers=1):
    """
    Return a dataloader with the data in the given directory.
    
    Args:
        data_dir: str, or list of str
            Directory/path to the data (see get_filenames() for the structure),
            or list of directories/paths to the subdirs.
        batch_size: int
            Number of samples to return as a batch.
        input_channels: str (default = "RG")
            Indicates the channels to load from the input images, e.g. "RG"
            for Red and Green.
        shuffle: bool (default = True)
            If True, the data is shuffled before being returned as batches.
        use_weights: bool (default = False)
            If True, will look for weight images and add them to the dataloaders.
        transform: callable (default = None)
            Transformation to apply to the input images.
        target_transform: callable (default = None)
            Transformation to apply to the target (and weight) images.
        num_workers: int (default = 1)
            Number of workers for the PyTorch Dataloader.
    
    Returns:
        A dataloader that generates tuples (input, target).
    """
    if use_weights:
        x, y, w = get_filenames(data_dir, use_weights=use_weights)
    else:
        x, y = get_filenames(data_dir)
        w = None
    dataset = ImageLoaderDataset(x, y, w, input_channels=input_channels,
                                 transform=transform, 
                                 target_transform=target_transform)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                           collate_fn=_pad_collate, num_workers=num_workers)


def get_all_dataloaders(data_dir, batch_size, input_channels="RG", test_dataloader=False,
                        use_weights=False,
                        synthetic_data=False, synthetic_ratio=None, synthetic_only=False,
                        train_transform=None, train_target_transform=None,
                        eval_transform=None, eval_target_transform=None):
    """
    Return a dataloader dictionary with the train, validation, and (optional) test dataloaders.
    
    Args:
        data_dir: str
            Directory/path to the data, it should contain "train/", "validation/",
            (optional) "test/", and (optional) "synthetic/" subdirs
            (see get_filenames() for their specific structure).
        batch_size: int
            Number of samples to return as a batch.
        input_channels: str (default = "RG")
            Indicates the channels to load from the input images, e.g. "RG"
            for Red and Green.
        test_dataloader: bool (default = False)
            If True, the dictionary will contain the test loader under "test".
        use_weights: bool (default = False)
            If True, will look for weight images and add them to the dataloaders.
        synthetic_data: bool (default = False)
            If True, the train loader will contain the synthetic data.
            See synthetic_ratio for choosing the proportion of synthetic data.
        synthetic_ratio: float (default = None)
            If synthetic_data is False, this is ignored.
            If not set, all data under "train/" and "synthetic/" are used for 
            the training (this is the default use).
            If set, it represents the ratio of synthetic vs. real data to use
            for training, and is based over real data size. For instance, if
            there are 1000 real frames and the synthetic_ratio is 25, enough
            real experiments will be taken to be as close as possible to 750
            frames, and enough synthetic experiments to be as close as 250 frames.
        synthetic_only: bool (default = False)
            If True, the train dataloader will contain only the synthetic data.
            As opposed to synthetic_ratio=1.0, this will use all of the data
            under "synthetic/", instead of using as many experiments as there 
            are in "train/". /!\ Overwrite synthetic_data.
        train_transform: callable (default = None)
            Transformation to apply to the train input images.
        train_target_transform: callable (default = None)
            Transformation to apply to the train target (and weight) images.
        eval_transform: callable (default = None)
            Transformation to apply to the validation/test input images.
        eval_target_transform: callable (default = None)
            Transformation to apply to the validation/test target (and weight) images.
    
    Returns:
        A dictionary with the train, validation and (optional) test dataloaders
        under the respective keys "train", "valid", and "test".
        Batches are made of (input, target) tuples.
    """
    # If synthetic data is used, build a list of folders for the train set
    if synthetic_only:
        train_dir = [os.path.join(data_dir, "synthetic/", subdir) for subdir 
                     in sorted(os.listdir(os.path.join(data_dir, "synthetic/")))]
    elif synthetic_data:
        if synthetic_ratio is None: # all of real and synthetic data
            train_dir = [os.path.join(data_dir, "train/", subdir) for subdir 
                         in sorted(os.listdir(os.path.join(data_dir, "train/")))] + \
                        [os.path.join(data_dir, "synthetic/", subdir) for subdir 
                         in sorted(os.listdir(os.path.join(data_dir, "synthetic/")))]
        else: # specific ratio between real and synthetic data
            # Load (in random order) real and synthetic folder names
            real_dirs = np.random.permutation(os.listdir(os.path.join(data_dir, "train/")))
            real_dirs = [os.path.join(data_dir, "train/", subdir) for subdir in real_dirs]
            synth_dirs = np.random.permutation(os.listdir(os.path.join(data_dir, "synthetic/")))
            synth_dirs = [os.path.join(data_dir, "synthetic/", subdir) for subdir in synth_dirs]
            
            # Create according array with frame quantity
            real_frames = np.array([len(os.listdir(os.path.join(real_dir, "rgb_frames/"))) 
                                    for real_dir in real_dirs])
            synth_frames = np.array([len(os.listdir(os.path.join(synth_dir, "rgb_frames/"))) 
                                     for synth_dir in synth_dirs])
            
            # If ratio is 0 or 1, take only real or synth
            if synthetic_ratio == 0.0:
                train_dir = real_dirs
            elif synthetic_ratio == 1.0: # take approximately as many frames as there is in real data
                n_tot_real_frames = np.sum(real_frames) # total number of frames in real data
                train_dir = synth_dirs
                n_synth_dirs = np.argmin(np.abs(np.cumsum(synth_frames) - n_tot_real_frames)) + 1
                
                train_dir = synth_dirs[:n_synth_dirs]
            else:            
                # Find number of real dirs and real synths to get good ratio of frames
                n_tot_real_frames = np.sum(real_frames) # total number of frames in real data
                n_real_dirs = np.argmin(np.abs(np.cumsum(real_frames) - \
                                               (1 - synthetic_ratio) * n_tot_real_frames)) + 1
                n_real_frames = np.sum(real_frames[:n_real_dirs]) # actual number of frames in real data
                # If not enough synthetic data, take all of it, and reduce real data accordingly
                n_synth_frames = np.sum(synth_frames)
                if n_synth_frames < synthetic_ratio / (1 - synthetic_ratio) * n_real_frames:
                    n_synth_dirs = len(synth_dirs)
                    n_real_dirs = np.argmin(np.abs(np.cumsum(real_frames) - \
                                                   (1 - synthetic_ratio) / synthetic_ratio * n_synth_frames)) + 1
                else:
                    n_synth_dirs = np.argmin(np.abs(np.cumsum(synth_frames) - \
                                             synthetic_ratio / (1 - synthetic_ratio) * n_real_frames)) + 1
                
                train_dir = real_dirs[:n_real_dirs] + synth_dirs[:n_synth_dirs]
    else:
        train_dir = os.path.join(data_dir, "train/")
    
    train_loader = get_dataloader(
            train_dir,
            batch_size=batch_size,
            input_channels=input_channels,
            shuffle=True,
            use_weights=use_weights,
            transform=train_transform,
            target_transform=train_target_transform
    )
    valid_loader = get_dataloader(
            os.path.join(data_dir, "validation/"),
            batch_size=batch_size,
            input_channels=input_channels,
            shuffle=False,
            use_weights=use_weights,
            transform=eval_transform,
            target_transform=eval_target_transform
    )
    if test_dataloader:
        test_loader = get_dataloader(
                os.path.join(data_dir, "test/"),
                batch_size=batch_size,
                input_channels=input_channels,
                shuffle=False,
                use_weights=use_weights,
                transform=eval_transform,
                target_transform=eval_target_transform
        )
        return {"train": train_loader, "valid": valid_loader, "test": test_loader}
    else:
        return {"train": train_loader, "valid": valid_loader}


## Image manipulations

def normalize_range(images):
    """Normalize the given float image(s) by changing the range from [0,1] to [-1,1]."""
    return images * 2.0 - 1.0


def make_images_valid(images):
    """Make sure the given images have correct value range and number of channels."""
    # Set range from [min,max] to [0,1]
    images = (images - images.min()) / (images.max() - images.min())
    # If only 2 channel (e.g. "RG"), add an empty third one
    if images.shape[1] == 2:
        shape = (images.shape[0], 1) + images.shape[2:]
        zero_padding = torch.zeros(shape, dtype=images.dtype).to(images.device)
        images = torch.cat([images, zero_padding], 1)
    return images