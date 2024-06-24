#!/bin/bash -i

### LSF syntax
#BSUB -nnodes 1             #number of nodes
#BSUB -W 4:00               #walltime in hours:minutes
#BSUB -e train_rom_single_intf_err.txt      #stderr
#BSUB -o train_rom_single_intf_out.txt      #stdout
#BSUB -J train_rom_single_intf          #name of job
#BSUB -q pbatch             #queue to use
#BSUB -G sosu               #account
###BSUB -Is bash              #interactive bash shell

### Shell scripting
date; hostname
echo -n '> JobID is '; echo $LSB_JOBID

echo -n '> Loading conda env'
#source ~/.bashrc
load_conda_env

# python/pytorch/horovod is forking a process,
# this setting is needed so that MPI will still work after a fork()
export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

# echo `which python`
# echo $HOSTNAME

# jsrun -r 1 python driver_train_rom.py --nx 480 --ny 24 --intr_ld 6 --intf_ld 4 --act_type Swish --intr_only
# jsrun -r 1 python driver_train_monolithic_rom.py --nx 240 --ny 12 --ld 4 --act_type Swish --dense_encoder --enc_hidden 3
# jsrun -r 1 python driver_train_monolithic_rom.py --nx 240 --ny 12 --ld 4 --act_type Swish


# Commands
# --------
python ./../scripts/driver_train_rom_single.py --nx 480 --ny 24 --intr_ld 12 --intf_ld 8 --act_type Swish --batch 64 --intf_only --gpu_idx 1 \
  --res_file ./../../../../run/data/residual/nx480_ny24_mu0.1_nsamples400.p \
  --snap_file ./../../../../run/data/snapshot/nx480_ny24_mu0.1_nsamples6400.p \
  --save_dir ./../../../../run/data2/

echo 'Done'
