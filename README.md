# AxoID
**AxoID**: automatic detection and tracking of axon identities over 2-photon neuroimaging data.  
It works in two main steps: 
  1. Detection of the regions of interest (ROIs) through deep learning (with PyTorch) as a binary segmentation
  2. Tracking of the ROI identities over the frames through custom losses and optimal assignment  
Finally, a GUI (written in PyQt5) is available for user corrections.


## Documentation
Documentation is available under [docs/](./docs), notably:
  * Installing AxoID: [Installation](./docs/Installation.md)
  * Explanation of the pipeline: [Pipeline](./docs/Pipeline.md)
  * How to run the main script, and its produced outputs: [Running AxoID](./docs/Running.md)
  * User correction with the GUI: [GUI](./docs/GUI.md)
  * Description of the `axoid` package: [Package](./docs/Package.md)
  * Overview of the different scripts: [Scripts](./docs/Scripts.md)
  * Overview of the different jupyter notebooks: [Notebooks](./docs/Notebooks.md)
  * Tracking using Coherent Point Drift: [CPD](./docs/CPD.md)


## Repository structure
  * `axoid`: main python package of *AxoID*
  * `docs`: documentation files
  * `images`: images used by the readmes
  * `model`: folder with the definition and weights of the network used for detection
  * `notebooks`: folder of Jupyter Notebooks used to test parts of AxoID
  * `scripts`: folder of python scripts, with `run_axoid.py`and `run_GUI.py` - the main scripts of *AxoID* and its GUI
  * `motion_compensation_path.py`: python file where the paths to the `motion_compensation` script and MATLAB should be reported if the user intend to have *AxoID* calls the optic flow warping (see [Installation](./docs/Installation.md))


## How to run
If you followed the installation instruction, you should have created a conda environment for *AxoID*. In order to run it, you must enter this environment `source activate axoid`, where `axoid` is the name you have given it. Every command below is assumed to be typed inside.

The main script can be launched as:
```
python scripts/run_axoid.py /path/to/experiment [--option VALUE]
```
or, if *AxoID* has been installed following [Installation](./docs/Installation.md), as:
```
axoid /path/to/experiment [--option VALUE]
```
where `/path/to/experiment` point to an experiment folder, excluding the `2Pimg/` (e.g.: `/data/lines/SS00001/2P/20190101/SS00001-tdTomGC6fopt-fly1/SS00001-tdTomGC6fopt-fly1-001/`).

Similarly, the user correction GUI can be launched using:
```
python scripts/run_GUI.py /path/to/experiment [--option VALUE]
```
or, if *AxoID* has been installed following [Installation](./docs/Installation.md), as:
```
axoid-gui /path/to/experiment [--option VALUE]
```
**Note:** if you run the GUI through `ssh`, you might need the option `-X` or `-Y` to forward the window to your screen. Additionally, note that it might be slower than running the GUI from your own machine.

### Optic flow warping
The main script works with data registered by Optic Flow Warping. It either looks for existing warped data, either tries to call a warping script.

If the user intend to use the automatic call to warping, it requires the [`motion_compensation`](https://github.com/NeLy-EPFL/motion_compensation) repository to be installed, and the paths in `motion_compensation_path.py` to be correctly set (see [Installation](./docs/Installation.md)).


## Features

### Train and optimize network
`axoid.detection.deeplearning` contains code to train and test deep networks for the ROI detection. Models were based on the U-Net, but this can be modified as long as it outputd the same format (an image with the same size as the input's, each pixel value is a logit).

### Create synthetic data
`axoid.detection.synthetic` focuses on creating synthetic 2-photon experimental data. It allows to generate stack of raw images with their according detection and identity ground truths.  
This was mostly used for training networks.

### Fine tuning
AxoID also explores fine tuning of the detection network on an experiment-by-experiment basis.  

This consists in selecting a subset of the experiment's frames, and generating ground truths (manually, or through some automated fashion) for them. Then, the network is further trained on these frames in order to increase its detection performance for this single experiment, and potentially help it find the correct axons.

However, note that such a fine tuned network tend to overfit said experiment. Therefore, it is recommended to first keep some annotated frames as a validation set to stop the training when validation performance does not increase, to detect and reduce overfitting of the training frames. Moreover, because the network will almost certainly overfit the experiment, it is only good for predicting this, and only this experiment. This process should then be repeated for each single experiment, or at least each fly.

### Internal model tracking 
The tracker framework is based on model matching: an *internal model* is created and updated based on the experimental frames, and each frame is then matched to it in order to identify the axons.

Here, the model represents a set of axon objects with some properties (as position on the image, shape, fluorescence intensity,...), and represents whate the tracker thinks the axons are, and how they are organized on the frame and relatively to each others.

This is sort of a general framework in which the matching between the model and the frames can be implemented arbitrarily. In this current work, it is based on optimal assignments with custom cost functions, but an idea based on point set registration was explored and could be developped (see [Coherent Point Drift](./docs/CPD.md)).

### Gui for correction and improvement
`axoid.GUI` implements an interface where the user can explore and correct the output of the main AxoID pipeline, made with PyQt5.
