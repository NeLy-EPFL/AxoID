# Synthetic data
Code for generating synthetic data that can be used for training the deep networks.  
The tests performed to develop these functions can be found under `notebooks/` (`synthetic_*.ipynb`).

# Pickled files
  * `GCaMP_kernel.pkl`: Tuple of numpy 1D-array of the GCaMP response vs time. The time steps are 1ms (the time between 2 values), meanwhile the value is in arbitrary units (originally taken in dF/F0 from (1)). The two array of the tuple are the synthetic response of (in order):
    * `kernel_f`: the fast indicator *6f*
    * `kernel_s`: the slow indicator *6s*
  * `stats_181121.pkl`: Pickle histograms representing the pixel intensity of real data in 256 bins (0->255). In order, the pickled objects are (in order):
    * `pixel_bkg`: histogram of background pixel intensity
    * `pixel_fg`: histogram of foreground (=ROI) pixel intensity
    * `roi_max`: histogram of maximal pixel intensity of ROIs
    * `roi_ave`: histogram of average pixel intensity of ROIs
    * `roi_med`: histogram of median pixel intensity of ROIs
    * `roi_q75`: histogram of 75th-percentile pixel intensity of ROIs
  * `stats_190221.pkl`: Pickle histograms representing the pixel intensity of real data in 256 bins (0->255). In order, the pickled objects are (in order):
    * `pixel_bkg`: histogram of background pixel intensity (both red and green channels)

(1) [Chen et. al, Ultrasensitive fluorescent proteins for imaging neuronal activity, *Nature*](https://www.nature.com/articles/nature12354#article-info)
