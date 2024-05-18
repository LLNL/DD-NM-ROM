# DD-NM-ROM
DD-NM-ROM integrates nonlinear-manifold reduced order models (NM-ROMs) with
domain decomposition (DD). NM-ROMs approximate the full order model (FOM) state
in a nonlinear-manifold by training a shallow, sparse autoencoder using FOM
snapshot data. These NM-ROMs can be advantageous over linear-subspace ROMs
(LS-ROMs) for problems with slowly decaying Kolmogorov-width. However, the
number of NM-ROM parameters that need to be trained scales with the size of the
FOM. Moreover, for ``extreme-scale" problems, the storage of high-dimensional
FOM snapshots alone can make ROM training expensive. To alleviate the training
cost, DD-NM-ROM applies DD to the FOM, computes NM-ROMs on each subdomain, and
couples them to obtain a global NM-ROM. This approach has several advantages:
Subdomain NM-ROMs can be trained in parallel, involve fewer parameters to be
trained than global NM-ROMs, require smaller subdomain FOM dimensional training
data, and can be tailored to subdomain-specific features of the FOM. The
shallow, sparse architecture of the autoencoder used in each subdomain NM-ROM
allows application of hyper-reduction (HR), reducing the complexity caused by
nonlinearity and yielding computational speedup of the NM-ROM. DD-NM-ROM
provides the first application of NM-ROM (with HR) to a DD problem. In
particular, DD-NM-ROM implements an algebraic DD reformulation of the FOM,
training a NM-ROM with HR for each subdomain, and a sequential quadratic
programming (SQP) solver to evaluate the coupled global NM-ROM.  The DD-NM-ROM
with HR approach is numerically compared to a DD-LS-ROM with HR on the 2D
steady-state Burgers' equation, showing an order of magnitude improvement in
accuracy of the proposed DD NM-ROM over the DD LS-ROM.


## Requirements
The code implementation was done using Python 3.9 and a CUDA-11.4 environment. Below are other required packages. 
- `numpy` (version 1.23.5)
- `scipy` (version 1.8.1)
- `matplotlib` (version 3.6.2)
- `PyTorch` (version 1.12.1)
- `torch-sparse` (version 0.6.10)
- `torch-scatter` (version 2.0.8)
- `dill` (version 0.3.6)
- [`sparselinear`](https://github.com/hyeon95y/SparseLinear) (version 0.0.5)
- `jupyter` (version 5.1.3)

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

## Citation
[Diaz, Alejandro N., Youngsoo Choi, and Matthias Heinkenschloss. "A fast and accurate domain decomposition nonlinear manifold reduced order model." Computer Methods in Applied Mechanics and Engineering 425 (2024): 116943.](https://doi.org/10.1016/j.cma.2024.116943)

## Authors 
- Alejandro Diaz
- Youngsoo Choi
- Matthias Heinkenschloss   

## Aknowledgement
A. Diaz was supported for this work by a Defense Science and Technology
Internship (DSTI) at Lawrence Livermore National Laboratory and a 2021 National
Defense Science and Engineering Graduate Fellowship, United States.  Y. Choi
was supported for this work by the US Department of Energy under the
Mathematical Multifaceted Integrated Capability Centers -- DoE Grant DE --
SC0023164; The Center for Hierarchical and Robust Modeling of Non-Equilibrium
Transport (CHaRMNET).

## License
DD-NM-ROM is distributed under the terms of the MIT license. All new contributions must be made under the MIT. See
[LICENSE-MIT](https://github.com/LLNL/DD-NM-ROM/blob/main/LICENSE)

LLNL Release Nubmer: LLNL-CODE-864366
