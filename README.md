# DD-NM-ROM
Author: Alejandro Diaz  
We apply LS-ROM and NM-ROM to the 2D Burgers equation.  
The code in this repo and its documentation is a work in progress. If you have any questions, please email me at and5@rice.edu.

## Requirements
The code implementation was done using Python 3.9 and a CUDA-11.4 environment. Below are other required packages. 
- `numpy`
- `scipy`
- `matplotlib`
- `PyTorch` (version 1.12.1)
- `torch-sparse` (version 0.6.10)
- `torch-scatter`
- `dill`
- [`sparselinear`](https://github.com/hyeon95y/SparseLinear) 
- `jupyter`

## Generating snapshot data for training
- To generate the training data, ssh into lassen and run `generate_training_data.sh`. This runs the scipts `generate_residual_data.py` and `generate_snapshot_data.py`.
- `generate_residual_data.py` and `generate_snapshot_data.py` take the following optional arguments
  * `--nx`:         number of FD grid points in x-direction
  * `--ny`:         number of FD grid points in y-direction
  * `--na`:         number of grid points in a-direction on parameter grid
  * `--nlam`:       number of grid points in lambda-direction on parameter grid
  * `--viscosity`:  viscosity parameter for 2D Burgers equation (fixed)
  * `--maxit`:      maximum number of iterations for Newton solver
  * `--tol`:        absolute tolerance for Newton solver
## Training a DD-LS-ROM (do this first)
- Run the notebook `driver_DD_LSROM.ipynb` in the CPU environment first on all configurations of interest.
- For example, run with different subdomain configurations, different number of training snapshots, etc. 
- Running on CPUs first lets you compute and store SVD data of snapshots, which then won't have to be recomputed in GPU environment. 

## Training a DD-NM-ROM
- Make sure the `driver_DD_LSROM.ipynb` notebook (see previous section) is run on the desired configuration using a CPU to store POD basis data. 
- Run `train_rom.sh.` This runs the python script `driver_train_rom.py`, which takes the following optional arguments 
  * `--nx`:                number of grid points in x-direction
  * `--ny`:                number of grid points in y-direction
  * `--n_sub_x`:           number of subdomains in x-direction
  * `--n_sub_y`:           number of subdomains in y-direction
  * `--intr_ld`:           ROM dimension for interior states
  * `--intf_ld`:           ROM dimension for interface states
  * `--intr_row_nnz`:      Number of nonzeros per row (col) in interior decoder (encoder) mask
  * `--intf_row_nnz`:      Row (col) shift in interface decoder (encoder) mask
  * `--intr_row_shift`:    Row (col) shift in interior decoder (encoder) mask
  * `--intf_row_shift`:    Number of nonzeros per row (col) in interface decoder (encoder) mask
  * `--nsnaps`:            Number of snapshots used for training
  * `--batch`:             batch size for training
  * `--intr_only`:         Only train autoencoders for interior states
  * `--intf_only`:         Only train autoencoders for interface states
  * `--act_type`:          Activation type. Only Sigmoid and Swish are implemented
- After `train_rom.sh` finishes, run the jupyter notebook `driver_DD_NMROM.ipynb`. 
- To train NM-ROMs using the strong ROM-port constraint formulation, run `train_port_decoders.sh` to train autoencoders for each port. This runs the script `driver_train_port_autoencoder.py`, which takes the same optional arguments as `driver_train_rom.py`.

## Generating Pareto fronts
- First run jobs `nmrom_nsnaps.sh`, `nmrom_sizes.sh` to train NM-ROMs using different numbers of snapshots and different ROM sizes, respectively. 
- Run the notebooks `driver_pareto_hr.ipynb`, `driver_pareto_nsnaps.ipynb`, `driver_pareto_romsize.ipynb`.
