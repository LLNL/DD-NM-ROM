# DD-NM-ROM

## Requirements
- To use the code in this repo, first create an Open-CE v1.7.2 with CUDA-11.4 for Lassen environment following the instructions [here](https://lc.llnl.gov/confluence/display/LC/2022/10/20/Open-CE+v1.7.2+with+CUDA-11.4+for+Lassen).  
- Install jupyter for the Open-CE using the instructions on the PowerAI page [here](https://lc.llnl.gov/confluence/display/LC/IBM+PowerAI+in+LC).
- I also recommend creating a virtual environment containing PyTorch that can be run on a CPU machine (e.g. quartz) ([instructions](https://lc.llnl.gov/confluence/display/LC/PyTorch+in+LC)).
For generating POD bases for the problem sizes considered in this repo, the SVD routine is much faster on CPUs. 
- Install jupter for ypur CPU PyTorch environment ([instructions](https://lc.llnl.gov/confluence/display/LC/JupyterHub+and+Jupyter+Notebook)). 

## Generating snapshot data for training
- To generate the training data, ssh into lassen and run generate_training_data.sh. This runs the scipts generate_residual_data.py and generate_snapshot_data.py.
- generate_residual_data.py and generate_snapshot_data.py take the following optional arguments 
  * --nx         number of FD grid points in x-direction
  * --ny         number of FD grid points in y-direction
  * --na         number of grid points in a-direction on parameter grid
  * --nlam       number of grid points in lambda-direction on parameter grid
  * --viscosity  viscosity parameter for 2D Burgers equation (fixed)
  * --maxit      maximum number of iterations for Newton solver
  * --tol        absolute tolerance for Newton solver

## Training a DD-LS-ROM (do this first)
- Run the notebook driver_DD_LSROM.ipynb in the CPU environment first on all configurations of interest (e.g. subdomain configurations). 
  * Running on CPUs first lets you compute and store SVD data of snapshots, which then won't have to be recomputed in GPU environment. 

## Training a DD-NM-ROM
- Make sure the driver_LSROM notebook (see previous section) is run on the desired configuration using a CPU to store POD basis data. 
- 
