#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module containing useful functions/classes for data manipulation with PyTorch.
Created on Mon Oct 22 13:54:19 2018

@author: nicolas
"""

import os
import numpy as np
import scipy.ndimage as ndi
from skimage import measure

from torch.utils import data

from axoid.utils.image import imread_to_float


class ImageLoaderDataset(data.Dataset):
    """Dataset that loads image online for efficient memory usage."""
    
    def __init__(self, x_filenames, y_filenames, w_filenames=None, 
                 input_channels="RGB", transform=None, target_transform=None):
        """
        Initialize the Dataset with the filenames and transforms.
        
        Parameters
        ----------
        x_filenames : list of str
            Contains the filenames/path to the input images.
        y_filenames : list of str
            Contains the filenames/path to the target images.
        w_filenames : list of str (optional)
            Contains the filenames/path to the weight images.
        input_channels : str (default = "RGB")
            Indicates the channels to load from the input images, e.g. "RG"
            for Red and Green, "R" for only Red.
        transform : callable (optional)
            Transformation to apply to the input images.
        target_transform : callable (optional)
            Transformation to apply to the target and weight images.
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
        """Load the files, and transform them before returning."""
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
    
    Note thate filenames should be ordered the same to identify correct tuples.
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
    data_dir can also be a list of the path to subdirs to use:
        data_dir = ["/path/to/subdir1", "/path/to/subdir2", ...]
    
    Parameters
    ----------
    data_dir : str, or list of str
        Directory/path to the data, or list of directories/paths to the subdirs.
    use_weights : bool (default = False)
        If True, will look for weight images and add them to the dataloaders.
    valid_extensions : tuple of str (default = ('.png', '.jpg', '.jpeg'))
        Tuple of the valid image extensions.
    
    Returns
    -------
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


def pad_collate(batch):
    """
    Collate function that pads input/target(/mask) images to the same size.
    
    Parameters
    ----------
    batch : list of tuple
        Each tuple contains input, target (and weight) of one element.
    
    Returns
    -------
    pad_batch : tensor
        Tensor of padded elements, where they all have the same size.
    """
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
                np.pad(item[0], [(0,0)] + padding, 'constant', constant_values=-1),
                np.pad(item[1], padding, 'constant', constant_values=0)))
        else:
            pad_batch.append((
                np.pad(item[0], [(0,0)] + padding, 'constant', constant_values=-1),
                np.pad(item[1], padding, 'constant', constant_values=0),
                np.pad(item[2], padding, 'constant', constant_values=0)))
    
    return data.dataloader.default_collate(pad_batch)


def get_dataloader(data_dir, batch_size, input_channels="RG",
                   shuffle=True,  use_weights=False,
                   transform=None, target_transform=None, num_workers=1):
    """
    Return a dataloader with the data in the given directory.
    
    Parameters
    ----------
    data_dir : str, or list of str
        Directory/path to the data (see get_filenames() for the structure),
        or list of directories/paths to the subdirs.
    batch_size : int
        Number of samples to return as a batch.
    input_channels : str (default = "RG")
        Indicates the channels to load from the input images, e.g. "RG"
        for Red and Green.
    shuffle : bool (default = True)
        If True, the data is shuffled before being returned as batches.
    use_weights : bool (default = False)
        If True, will look for weight images and add them to the dataloaders.
    transform : callable (default = None)
        Transformation to apply to the input images.
    target_transform : callable (default = None)
        Transformation to apply to the target (and weight) images.
    num_workers : int (default = 1)
        Number of workers for the PyTorch Dataloader.
    
    Returns
    -------
    A dataloader that generates tensors of batches.
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
                           collate_fn=pad_collate, num_workers=num_workers)


def get_all_dataloaders(data_dir, batch_size, input_channels="RG", test_dataloader=False,
                        use_weights=False,
                        synthetic_data=False, synthetic_ratio=None, synthetic_only=False,
                        train_transform=None, train_target_transform=None,
                        eval_transform=None, eval_target_transform=None):
    """
    Return a dataloader dictionary with the train, validation, and (optional) test dataloaders.
    
    Parameters
    ----------
    data_dir : str
        Directory/path to the data, it should contain "train/", "validation/",
        (optional) "test/", and (optional) "synthetic/" subdirs
        (see get_filenames() for their specific structure).
    batch_size : int
        Number of samples to return as a batch.
    input_channels : str (default = "RG")
        Indicates the channels to load from the input images, e.g. "RG"
        for Red and Green.
    test_dataloader : bool (default = False)
        If True, the dictionary will contain the test loader under "test".
    use_weights : bool (default = False)
        If True, will look for weight images and add them to the dataloaders.
    synthetic_data : bool (default = False)
        If True, the train loader will contain the synthetic data.
        See synthetic_ratio for choosing the proportion of synthetic data.
    synthetic_ratio : float (default = None)
        If synthetic_data is False, this is ignored.
        If not set, all data under "train/" and "synthetic/" are used for 
        the training (this is the default use).
        If set, it represents the ratio of synthetic vs. real data to use
        for training, and is based over real data size. For instance, if
        there are 1000 real frames and the synthetic_ratio is 25, enough
        real experiments will be taken to be as close as possible to 750
        frames, and enough synthetic experiments to be as close as 250 frames.
    synthetic_only : bool (default = False)
        If True, the train dataloader will contain only the synthetic data.
        As opposed to synthetic_ratio=1.0, this will use all of the data
        under "synthetic/", instead of using as many experiments as there 
        are in "train/". /!\ Overwrite synthetic_data.
    train_transform : callable (default = None)
        Transformation to apply to the train input images.
    train_target_transform : callable (default = None)
        Transformation to apply to the train target (and weight) images.
    eval_transform : callable (default = None)
        Transformation to apply to the validation/test input images.
    eval_target_transform : callable (default = None)
        Transformation to apply to the validation/test target (and weight) images.
    
    Returns
    -------
    A dictionary with the train, validation and (optional) test dataloaders
    under the respective keys "train", "valid", and "test".
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

# Following transform is to avoid 2x2 maxpoolings on odd-sized images
# (it makes sure down- and up-sizing are consistent throughout the network)
def pad_transform(image, u_depth):
    """
    Pad the image to assure its height and width are mutliple of 2**u_depth.
    
    If RGB, format should be channels first (<=> CHW).
    
    Parameters
    ----------
    image : ndarray
        Image array in numpy.
    u_depth : int
        Depth of the U-Net network.
    
    Returns
    -------
    padded_image : ndarray
        Image padded with zeros to a size that can be divided by 2**u_depth.
    """
    factor = 2 ** u_depth
    if image.ndim == 3: # channels first
        height, width = image.shape[1:]
    elif image.ndim == 2:
        height, width = image.shape
        
    # Do nothing if image has correct shape
    if height % factor == 0 and width % factor == 0:
        return image
    
    height_pad = (factor - height % factor) * bool(height % factor)
    width_pad = (factor - width % factor) * bool(width % factor)
    padding = [(int(np.floor(height_pad/2)), int(np.ceil(height_pad/2))), 
               (int(np.floor(width_pad/2)), int(np.ceil(width_pad/2)))]
    if image.ndim == 3: # do not pad channels
        return np.pad(image, [(0,0)] + padding, 'constant')
    elif image.ndim == 2:
        return np.pad(image, padding, 'constant')

def pad_transform_stack(stack, u_depth):
    """
    Pad the stack to assure its height and width are mutliple of 2**u_depth.
    
    If RGB, format should be channels first (<=> NCHW).
    
    Same as pad_transform(), but for stack of images.
    """
    # Loop over the image and calls pad_transform()
    # This is not optimal, but still fast enough
    pad_stack = []
    for i in range(len(stack)):
        pad_stack.append(pad_transform(stack[i], u_depth))
    return np.stack(pad_stack)

def _contour_weights(image):
    """Compute the pixel-wise weighting of ROI contours."""
    distances = ndi.distance_transform_edt(1 - image)
    weights = np.exp(- (distances / 3.0) ** 2) - image
    return weights    
def _separation_weights(image):
    """Compute the pixel-wise weighting of separation between close ROIs."""
    labels, num = measure.label(image, connectivity=1, return_num=True)
    # If less than 2 ROIs, cannot have any separation border
    if num < 2:
        return 0
    distances = np.zeros(image.shape + (num,), np.float32)
    for i in range(0, num):
        distances[...,i] = ndi.distance_transform_edt(labels != (i+1))
    distances = np.sort(distances)
    weights = np.exp(-((distances[...,0] + distances[...,1]) / 6) ** 2) * (1 - image)
    return weights    
def compute_weights(image, contour=True, separation=True):
    """
    Return the pixel-wise weighting of the binary image/stack.
    
    Not that these weights will be rescaled by the positive and negative
    weighting at train time as: weight = neg_w + (pos_w - neg_w) * weight.
    Therefore, they adapt to the proportion of positive pixels automatically.
    
    Parameters
    ----------
    image : ndarray
        Image or stack of image (binary) corresponding to the targetted 
        segmentation(s) of a frame(s).
    contour : bool (default = True)
        If True, weights will be increased around ROIs.
        This is useful to have a gradient around ROIs instead of a step, and 
        helps to learn contours.
    separation : bool (default = True)
        If True, weights will be increased between close ROIs.
        This is usefull to increase importance of separation borders, and help
        to learn to draw individual ROI.
    
    Returns
    -------
    weights : ndarray
        Image(s) where the greyscale value corresponds to the pixels' weights.
    """
    weights = np.zeros(image.shape, np.float32)
    image = image.astype(np.bool)
    if not contour and not separation:
        return weights
    if weights.ndim == 3:
        for i in range(len(image)):
            if contour:
                weights[i] += _contour_weights(image[i])
            if separation:
                weights[i] += _separation_weights(image[i])
    else:
        if contour:
            weights += _contour_weights(image)
        if separation:
            weights += _separation_weights(image)
    return weights
