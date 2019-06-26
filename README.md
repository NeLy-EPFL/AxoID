# AxoID
**AxoID**: automatic detection and tracking of axon identities over 2-photon neuroimaging data.  
It works in two main steps: 
  1. Detection of the regions of interest (ROIs) through deep learning (with PyTorch) as a binary segmentation
  2. Tracking of the ROI identities over the frames through custom losses and optimal assignment
Finally, a GUI (written in PyQt5) is available for user corrections.


## Documentation
Documentation is available under [docs](./docs), notably:
  * Installing AxoID: [Installation](./docs/Installation.md)
  * Explanation of the pipeline: [Pipeline](./docs/Pipeline.md)
  * How to run the main script: [Running AxoID](./docs/Running Axoid.md)
  * User correction with the GUI: [GUI](./docs/GUI.md)
  * Description of the `axoid` package: [Package](./docs/Package.md)
  * Overview of the different scripts: [Scripts](./docs/Scripts.md)
  * Overview of the different jupyter notebooks: [Notebooks](./docs/Notebooks.md)
  * Tracking using Coherent Point Drift: [CPD](./docs/CPD.md)


## Repository structure
  * `axoid`: main python package of *AxoID*
  * `model`: folder with the definition and weights of the network used for detection
  * `notebooks`: folder of Jupyter Notebooks used to test parts of AxoID
  * `scripts`: folder of python scripts, with `run_axoid.py`and `run_GUI.py` - the main scripts of *AxoID* and its GUI
  * `motion_compensation_path.py`: python file where the paths to the `motion_compensation` script and MATLAB should be reported if the user intend to have *AxoID* calls the optic flow warping (see [Installation](./docs/Installation.md))


## How to run
If you followed the installation instruction, you should have created a conda environment for *AxoID*. In order to run it, you must enter this environment `source activate axoid`, where `axoid` is the name you have given this environment. Every command below is assumed to be typed in it. 

The main script can be launched as:
```
python run_axoid.py /path/to/experiment [--option VALUE]
```
or, if *AxoID* has been installed following [Installation](./docs/Installation.md), as:
```
axoid /path/to/experiment [--option VALUE]
```
where `/path/to/experiment` point to an experiment folder, excluding the `2Pimg/` (e.g.: `/data/lines/SS00001/2P/20190101/SS00001-tdTomGC6fopt-fly1/SS00001-tdTomGC6fopt-fly1-001/`).

Similarly, the user correction GUI can be launched using:
```
python run_GUI.py /path/to/experiment [--option VALUE]
```
or, if *AxoID* has been installed following [Installation](./docs/Installation.md), as:
```
axoid-gui /path/to/experiment [--option VALUE]
```

### Optic flow warping
The main script works with data registered by Optic Flow Warping. It either looks for existing warped data, either try to call a warping script.

If the user intend to use the automatic call to warping, it requires the [`motion_compensation`](https://github.com/NeLy-EPFL/motion_compensation) repository to be installed, and the paths in `motion_compensation_path.py` to be correctly set (see [Installation](./docs/Installation.md)).
