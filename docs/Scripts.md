## Python scripts
Under [scripts/](../scripts), multiple python scripts are available.  
The main ones are:
  * `run_axoid.py`: main script to run the *AxoID* pipeline on an experimental folder, see [Running AxoID](./Running.md) for details on how to use. This script only calls the main function in `axoid.main`.
  * `run_GUI.py`: main script to run the *AxoID* user correction GUI on an experimental folder, see [GUI](./GUI.md) for details on how to use. This script only calls the main function in `axoid.GUI.main`.

The other scripts were used during the development of *AxoID*:
  * `generate_weights.py`: generate the pixel-wise weight images of all experiments in the folder, or of all datasets in the folder written in the code
  * `run_generation.py`: generate a dataset of multiple synthetic experiments. Parameters as number of stacks and images per stacks have to be set inside the code
  * `run_gridsearch.py`: perform a gridsearch over hyperparameters set in the code. It calls `run_train.py` to perform individual training.
  * `run_train.py`: fully train a network, and accept different command line options
