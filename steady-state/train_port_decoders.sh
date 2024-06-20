#BSUB -nnodes 1
#BSUB -W 12:00
#BSUB -q pbatch
#BSUB -J ports_ld1
#BSUB -o port_ld1.txt

# python/pytorch/horovod is forking a process,
# this setting is needed so that MPI will still work after a fork()
export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

echo $HOSTNAME

# train DD-NM-ROMs for different ROM sizes
jsrun -r 1 python driver_train_port_autoencoder.py --port_ld 2 --port_row_nnz 3 --port_row_shift 3 --nx 480 --ny 24 --act_type Sigmoid