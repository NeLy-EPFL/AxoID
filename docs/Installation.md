## Installation
The following installation steps assumes that Anaconda is installed. *AxoID* was only tested on Ubuntu.  

First, download the repository:
```
git clone https://github.com/NeLy-EPFL/AxoID
cd AxoID
```

Then, create an environment with Python 3.6:
```
conda create -n axoid python=3.6
```
where `axoid` will be the name of the conda environment (you can choose any name).  
Then, you have to enter this environment (you will have to enter it each time you want to use *AxoID*) with
```
source activate axoid
```
or the name you have given it (In order to quit the environment, you can use `conda deactivate`). The command line should have a `(axoid)` at the beginning to show that you are in it.

If you intend to let *AxoID* call the optic flow warping, you will need the [`motion_compensation`](https://github.com/NeLy-EPFL/motion_compensation) repository.  
Follow the instructions to install it. Once it is installed and working, you will need to edit the paths in `motion_compensation_path.py` to the motion_compensation script, and MATLAB release.  
Without this, *AxoID* will not be able to warp the data, and it is the user's responsability to do so beforehand.

If you intend to use the Jupyter Notebooks, you need to install it:
```
conda install jupyter
```

Finally, you can install the `axoid` package and its dependencies with (be sure to be in the AxoID/ folder)
```
pip install -e .
```

### Installed commands
Inside the new conda environment just created, the `axoid` package should have been installed and therefore can be imported in Python (using `import axoid`).  
Additionally, two new commands were added to run easily *AxoID* and its GUI:
```
axoid /path/to/experiment [--option VALUE]
```
to run the main pipeline on the experiment, and
```
axoid-gui /path/to/experiment [--option VALUE]
```
to run the user correction GUI on the experiment, after it was processed by *AxoID*.

For more information, see [Running AxoID](./Running AxoId.md)
