#!/bin/bash -i

### LSF syntax
#BSUB -nnodes 1             #number of nodes
#BSUB -W 4:00               #walltime in hours:minutes
#BSUB -e train_rom_port_err.txt      #stderr
#BSUB -o train_rom_port_out.txt      #stdout
#BSUB -J train_rom_port          #name of job
#BSUB -q pbatch             #queue to use
###BSUB -G sosu               #account
###BSUB -Is bash              #interactive bash shell

### Shell scripting
date; hostname
echo -n '> JobID is '; echo $LSB_JOBID

echo -n '> Loading conda env';
#source ~/.bashrc
load_conda_env

# python/pytorch/horovod is forking a process,
# this setting is needed so that MPI will still work after a fork()
export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

# train DD-NM-ROMs for different ROM sizes
# jsrun -r 1 python driver_train_port_autoencoder.py --port_ld 2 --port_row_nnz 3 --port_row_shift 3 --nx 480 --ny 24 --act_type Sigmoid


# Commands
# --------
python ./../scripts/driver_train_rom_port.py --nx 480 --ny 24 --port_ld 2 --port_row_nnz 3 --port_row_shift 3 --act_type Sigmoid \
  --res_file ./../../../../run/data/residual/nx480_ny24_mu0.1_nsamples400.p \
  --snap_file ./../../../../run/data/snapshot/nx480_ny24_mu0.1_nsamples6400.p \
  --save_dir ./../../../../run/data/
  
echo 'Done'


python driver_train_rom_port.py --nx 480 --ny 24 --port_ld 2 --port_row_nnz 3 --port_row_shift 3 --act_type Sigmoid \
  --res_file ./../../../../run/data/residual/nx480_ny24_mu0.1_nsamples400.p \
  --snap_file ./../../../../run/data/snapshot/nx480_ny24_mu0.1_nsamples6400.p \
  --save_dir ./../../../../run/data2/