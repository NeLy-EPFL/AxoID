# Synthetic data
Code for testing and generating synthetic data that can be used for training the deep networks.

# Files
  * `run_generation.py`: script used to create the synthetic data (parameters should be changed inside the code)
  * `stats_181121.pkl`: Pickle histograms representing the pixel intensity of real data in 256 bins (0->255). In order, the pickled objects are:
    * `pixel_bkg`: histogram of background pixel intensity
    * `pixel_fg`: histogram of foreground (=ROI) pixel intensity
    * `roi_max`: histogram of maximal pixel intensity of ROIs
    * `roi_ave`: histogram of average pixel intensity of ROIs
    * `roi_med`: histogram of median pixel intensity of ROIs
    * `roi_q75`: histogram of 75th-percentile pixel intensity of ROIs
  * `stats_190221.pkl`: Pickle histograms representing the pixel intensity of real data in 256 bins (0->255). In order, the pickled objects are:
    * `pixel_bkg`: histogram of background pixel intensity (both red and green channels)
    * `roi_max`: histogram of maximal pixel intensity of ROIs (both red and green channels)
  * `synthetic_generation.ipynb`: notebook used to test the generation of multiple synthetic experiments
  * `synthetic_tests.ipynb`: notebook used to test the synthetic data generation pipeline
  * `utils_common`: link to the common utils library
