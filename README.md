# AxoID
**AxoID**: automatic detection and tracking of axon identities from 2-photon microscopy neural data.  
AxoID works in two main steps: 
  1. Detect Regions-of-Interest (ROIs) via deep learning (using PyTorch) as a binary segmentation.
  2. Track ROI identities across imaging frames using custom losses and optimal assignment.
Finally, a GUI written in PyQt5 is available to perform manual corrections.


## Documentation
Documentation is available under [docs/](./docs), specifically:
  * Installing AxoID: [Installation](./docs/Installation.md)
  * Explanation of the pipeline: [Pipeline](./docs/Pipeline.md)
  * How to run the main script, and its outputs: [Running AxoID](./docs/Running.md)
  * Manual correction using the GUI: [GUI](./docs/GUI.md)
  * Description of the `Axoid` package: [Package](./docs/Package.md)
  * Overview of the various scripts: [Scripts](./docs/Scripts.md)
  * Overview of the various Jupyter notebooks: [Notebooks](./docs/Notebooks.md)
  * Tracking using Coherent Point Drift: [CPD](./docs/CPD.md)


## Repository structure
  * `axoid`: main python package of *AxoID*
  * `docs`: documentation files
  * `images`: images used by the readmes
  * `model`: folder with the definition and weights of the network used for detection
  * `notebooks`: folder of Jupyter Notebooks used to test parts of AxoID
  * `scripts`: folder of Python scripts, with `run_axoid.py`and `run_GUI.py` - the main scripts of *AxoID* and its GUI
  * `motion_compensation_path.py`: Python file where the paths to the `motion_compensation` script and MATLAB should be reported if the user intends to have *AxoID* call optic flow warping (see [Installation](./docs/Installation.md))


## How to run
If you have followed the installation instructions, you should have created a conda environment for *AxoID*. Now, to run it, you must enter this environment `source activate axoid`, where `axoid` is the name you have given it. Every command below is assumed to be typed inside this environment.

The main script can be launched as:
```
python scripts/run_axoid.py /path/to/experiment [--option VALUE]
```
or, if *AxoID* has been installed following [Installation](./docs/Installation.md), as:
```
axoid /path/to/experiment [--option VALUE]
```
where `/path/to/experiment` points to an experiment folder, excluding the `2Pimg/` (e.g.: `/data/lines/SS00001/2P/20190101/SS00001-tdTomGC6fopt-fly1/SS00001-tdTomGC6fopt-fly1-001/`).

Similarly, the user correction GUI can be launched using:
```
python scripts/run_GUI.py /path/to/experiment [--option VALUE]
```
or, if *AxoID* has been installed following [Installation](./docs/Installation.md), as:
```
axoid-gui /path/to/experiment [--option VALUE]
```
**Note:** if you ran the GUI using `ssh`, you might need the option `-X` or `-Y` to forward the window to your screen. Additionally, note that it might be slower than running the GUI from your own machine.

### Optic flow warping
The main script works with data registered using Optic Flow Warping. It either looks for existing warped data, or tries to call the warping script.

If the user intends to use the automatic call to warping, the [`motion_compensation`](https://github.com/NeLy-EPFL/motion_compensation) repository must be installed, and the paths in `motion_compensation_path.py` must be correctly set (see [Installation](./docs/Installation.md)).


## Features

### Train and optimize the network
`axoid.detection.deeplearning` contains code needed to train and test deep networks for ROI detection. Models were based on a U-Net, but this can be modified as long as it outputs the same format (an image with the same size as the input's, each pixel value is a logit).

### Create synthetic data
`axoid.detection.synthetic` focuses on creating synthetic 2-photon experimental data. It generates a stack of raw images with their associated detection and identity ground truths.
This was mostly used to train networks.

### Fine-tuning
AxoID also permits fine-tuning of the detection network on an experiment-by-experiment basis.  

This consists of selecting a subset of the experimental frames, and generating ground truths (manually, or in an automated manner) for them. Then, the network is further trained on these frames to increase its detection performance for this single experiment, and potentially to help it find the correct axons.

However, note that such a fine-tuned network tends to overfit the particular experiment. Therefore, it is recommended to first keep some annotated frames as a validation set to stop the training when validation performance does not increase. This allows one to detect and reduce overfitting on the training frames. Moreover, because the network will almost certainly overfit, it is only good for predicting this particular experiment. This process should then be repeated for other individual experiments, or at least each data from each individual fly.

### Internal model tracking 
The tracker framework is based on model matching: an *internal model* is created and updated based on the experimental frames, and each frame is then matched to it to identify the axons.

Here, the model represents a set of axon objects with some properties (e.g., position on the image, shape, fluorescence intensity,...), and represents what the tracker thinks the axons are, and how they are organized on the frame with respect to one another.

This is a general framework in which matching between the model and the frames can be implemented arbitrarily. In this work, it is based on optimal assignments with custom cost functions, but an idea based on point set registration was explored and could be developped (see [Coherent Point Drift](./docs/CPD.md)).

### Gui for correction and improvement
`axoid.GUI` implements an interface where the user can explore and correct the output of the main AxoID pipeline, made using PyQt5.
