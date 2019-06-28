## How to run AxoID
First, enter the conda environment in which you have installed *AxoID*. 

The main script can be launched as:
```
axoid /path/to/experiment [--option VALUE]
```
where `/path/to/experiment` point to an experiment folder, excluding the `2Pimg/` (e.g.: `/data/lines/SS00001/2P/20190101/SS00001-tdTomGC6fopt-fly1/SS00001-tdTomGC6fopt-fly1-001/`).

Similarly, the user correction GUI can be launched using:
```
axoid-gui /path/to/experiment [--option VALUE]
```
**Note:** if you run the GUI through `ssh`, you might need the option `-X` or `-Y` to forward the window to your screen. Additionally, note that it might be slower than running the GUI from your own machine.

Both scripts accept the argument `--help` for a list of their optional arguments.


### Outputs
In this part, the folder with the experimental data given to *AxoID* is referred to as `exp/`, and should contain `2Pimg/`, and potentially `output/`.  

First, if they do not already exists, it will created images and image stacks in `2Pimg/`:
  * `warped_RGB.tif`: image stack with the optic flow warped data
  * `AVG_warped_RGB.tif`: temporal average of `warped_RGB.tif`
  * `AVG_warped_RGB.tif`: temporal average of `warped_RGB.tif` tdTomato channel
  * `ccreg_RGB.tif`: image stack with the cross-correlation registered data
  * `AVG_ccreg_RGB.tif`: temporal average of `ccreg_RGB.tif`
  * `AVG_ccreg_RGB.tif`: temporal average of `ccreg_RGB.tif` tdTomato channel

Then, it will create 3 folders (if not already existing):
  * `axoid_internal/`: all the data internal to *AxoID* will be stored there, for debugging purposes, and for the GUI
  * `GC6_auto/`: folder storing the fluorescence traces, with the same structure as before
  * `ROI_auto/`: folder containing an image stack with the ROI contours overlayed over the raw data

In each of these folder, *AxoID* actually creates 3 subfolder in which it stores the same kind of data, but for different inputs:
  * `raw/`: contains the results and outputs of the *AxoID* pipeline applied to the raw data
  * `ccreg/`: contains the results and outputs of the *AxoID* pipeline applied to the cross-correlation registered data
  * `warped/`: contains the results and outputs of the *AxoID* pipeline applied to the optic flow warped data

In the `axoid_internal` folder, *AxoID* will saved the following outputs (note that `raw/` might not have them all as its pipeline is slightly simplified, see [Pipeline](./Pipeline.md):
  * `cuts.pkl`: the cuts automatically applied to the model, as python pickled object
  * `identities.tif`: the final identity frames
  * `identities_precut.tif`: the identity frames before the automatic cuts were applied
  * `indices`:
  * `indices_init.txt`: text file with the indices of the frames used as similar frames for the projection and fine tuning. If no fine tuning (for `raw/`), it is the index of the frame used for initializing the tracker model
  * `input.tif`: image stack of the input data, be it the raw, cross-correlation registered, or warped data
  * `input_fluo.tif`: image stack used for extracting the fluorescence in the **warped** data (in opposite to `input.tif` which were used to find and track ROIs)
  * `model.tif`: image of the final tracker model with the different axons and their identities
  * `model_precut`: image of the tracker model with the different axons and their identities, before automatic cuts
  * `rgb_init.tif`: RGB image used to initialize the tracker model (either the temporal projection of the frame used for fine tuning, either a selected frame if no fine tuning)
  * `seg_init.tif`: binary segmentation of `rgb_init.tif`, used to initialize the tracker model
  * `seg_init_cut.tif`: binary segmentation of `rgb_init.tif` with cuts, used to automatically find cuts
  * `segmentations.tif`: stack of binary image corresponding to the ROI detections
