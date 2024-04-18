#BSUB -nnodes 1
#BSUB -W 2:00
#BSUB -q pbatch
#BSUB -J conf2x1_intf3
#BSUB -o conf2x1_intf3.txt

# python/pytorch/horovod is forking a process,
# this setting is needed so that MPI will still work after a fork()
export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

echo $HOSTNAME

# train DD-NM-ROMs for different DD configurations
# jsrun -r 1 python driver_train_port_autoencoder.py --n_sub_x 2 --n_sub_y 1 --nx 480 --ny 24 --act_type Sigmoid
# 
jsrun -r 1 python driver_train_rom.py --n_sub_x 2 --n_sub_y 1 --intr_ld 6 --intf_ld 3 --nx 480 --ny 24 --act_type Swish --intf_only