## Jupyter Notebooks
In [notebooks/](../notebooks), there are multiple Jupyter Notebooks used to test the different parts of *AxoID* during its development.  
Note that they might not all be up-to-date and mostly contain testing bits of codes. The code in the `axoid` package should be the last working version.

**Note:** on most notebooks, they might look for model in a ./data/models/ folder not existing on the repository because it was too heavy. If it is the case, you can use the model in ./model/ instead, or re-create a ./data/models/ folder.

  * `AxoID_warp.ipynp`: test of the entire pipeline on warped data, with automatic fine tuning
  * `ROI_detection_DL.ipynp`: train networks, and visualize their test performance
  * `annotate_experiments.ipynp`: use many methods to try and create annotations for full experiments (annotations are the ground truth binary detection)
  * `fine_tuning_test.ipynp`: fine tune the network on a few manually annotated frames (annotated outside the notebook), and test results over the full experiment
  * `synthetic_generation.ipynp`: test the functions to generate full stacks of synthetic data (see `axoid.detection.synthetic`)
  * `synthetic_tests.ipynp`: test the different steps of synthetic image generation in order to see how to produce realistic looking images
  * `tracking_test.ipynp`: test the internal model idea for the tracker. The Coherent Point Drift algorithm is also tested at the very bottom.
