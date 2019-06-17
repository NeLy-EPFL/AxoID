# README NOT UP-TO-DATE (ONGOING)
# AxoID
*AxoID*: automatic detection and tracking of axon identities over 2-photon neuroimaging data.  
It works in two steps: first a detection of Regions of interest (ROIs) as a binary segmentation, then a tracking of ROIs' identities over the frames. Finally, a GUI is available for user corrections.

# Repository structure
  * `axoid`: code of the main python package *AxoID*
  * `notebooks`: folder of Jupyter Notebooks used to test parts of AxoID
  * `scripts`: folder of python scripts, mainly containing `run_axoid.py` - the main script of *AxoID*
  * `motion_compensation_path.py`: python file where the paths to `motion_compensation` script and MATLAB should be reported if the user intend to have *AxoID* calls the optic flow warping

# How to run
The main script can be launched as:
```
python run_axoid.py /path/to/experiment [--option VALUE]
```
where `/path/to/experiment` point to an experiment folder, excluding the `2Pimg/` (e.g.: `/data/lines/SS00001/2P/20190101/SS00001-tdTomGC6fopt-fly1/SS00001-tdTomGC6fopt-fly1-001/`).

### Optic flow warping
The script requires either available warped data, either the `motion_compensation` repository installed inoreder to automatically call the warping.  
If the user intend to use `motion_compensation`, the file `motion_compensation_path.py` should be modified accordingly by writing the paths to the main script of `motion_compensation` and to MATLAB.

### Optional arguments
To print the help (see below), use `python run_axoid.py --help`. The following message will print:
```
usage: run_axoid.py [-h] [--force_ccreg] [--force_warp] [--maxiter MAXITER]
                    [--no_gpu] [-s SEED] [-t] [-v] [--warpdir WARPDIR]
                    [--warp_g WARP_G] [--warp_l WARP_L]
                    experiment

Main script of AxoID that detects and tracks ROIs on 2-photon neuroimaging
data.

positional arguments:
  experiment            path to the experiment folder (excluding "2Pimg/")

optional arguments:
  -h, --help            show this help message and exit
  --force_ccreg         force initial cross-correlation registration of the
                        raw data, even if "ccreg_RGB.tif" already exists
  --force_warp          force initial optic flow warping of the raw data, even
                        if "warped_RGB.tif" already exists
  --maxiter MAXITER     maximum number of iteration for the network fine
                        tuning (default=1000)
  --no_gpu              disable GPU utilization
  -s SEED, --seed SEED  seed for the script (default=1)
  -t, --timeit          time the script. If --verbose is used, sub-parts of
                        the script will also be timed.
  -v, --verbose         enable output verbosity
  --warpdir WARPDIR     directory where the warped output is stored inside
                        2Pimg/
  --warp_g WARP_G       gamma parameter for the optic flow warping
                        (default=10)
  --warp_l WARP_L       lambda parameter for the optic flow warping
                        (default=300)
```
