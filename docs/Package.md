## `axoid` package
### File structure
```
axoid/
├── GUI/						user correction gui
│   ├── annotation.py			page for manual annotation *(not implemented)*
│   ├── constants.py			constants and common parameters for the whole GUI
│   ├── correction.py			page for frame-wise correction *(can only change identity of ROIs)*
│   ├── image.py				image widgets for easy image and stack display in PyQt5
│   ├── main.py					main window and function for the gui
│   ├── model.py				page for model correction
│   ├── multipage.py			page widgets for easy multi-pages app in PyQt5
│   └── selection.py			page for output selection
│
├── detection/					ROI detection as binary segmentation
│   ├── cv/						detector using computer vision approach
│   │   └── detector.py			main function for detecting ROI with computer vision
│   │
│   ├── deeplearning/			detector using deep learning approach in PyTorch
│   │   ├── data.py				data loaders and manipulation
│   │   ├── finetuning.py		fine tune existing network and manually annotated frames with OpenCV
│   │   ├── loss.py				create loss function
│   │   ├── metric.py			create metric functions
│   │   ├── model.py			define, save and load network models
│   │   ├── test.py				test a trained model
│   │   └── train.py			train a model
│   │
│   ├── synthetic/				generate synthetic data
│   │   ├── GCaMP_kernel.pkl	synthetic kernels of GCaMP response to convolve with spike trains
│   │   ├── generation.py		functions to generate synthetic experiments
│   │   ├── stats_181121.pkl	histograms representing the pixel intensity of real data
│   │   └── stats_190221.pkl	same as above, updated version (see README.md for more details)
│   └── clustering.py			find cluster of similar frames for automatic finetuning
│
├── tracking/					ROI identity tracking through frames
│   ├── **pycpd/**				coherent point drift python implementation
│   │							(see [Coherent Point Drift](./CPD.md), and pycpd/README.md)
│   ├── cutting.py				functions for "cutting" ROIs in two sub-ROIs
│   ├── model.py				internal model tracker using custom cost functions
│   └── utils.py 				general useful functions for tracking ROIs
│
├── utils/						general useful functions and classes
│   ├── ccreg.py				image registration using cross-correlation (see Guizar M. 2008)
│   ├── fluorescence.py			compute, extract and save fluorescence
│   ├── image.py				load and save images, create overlays
│   ├── metrics.py				different metric functions for binary images
│   ├── multithreading.py		run code/functions in parallel
│   ├── processing.py			processing of images/stacks of images
│   └── script.py				scripts and using-scripts related module
│
└── main.py						main pipeline to apply to single experiment
```
