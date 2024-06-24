#BSUB -nnodes 1
#BSUB -W 12:00
#BSUB -q pbatch
#BSUB -J romsize
#BSUB -o romsize_ports.txt

# python/pytorch/horovod is forking a process,
# this setting is needed so that MPI will still work after a fork()
export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

echo $HOSTNAME

# train DD-NM-ROMs for different ROM sizes
jsrun -r 1 python driver_train_rom.py --intr_ld 4 --intf_ld 2 --nx 480 --ny 24 --act_type Swish
jsrun -r 1 python driver_train_rom.py --intr_ld 6 --intf_ld 3 --nx 480 --ny 24 --act_type Swish
jsrun -r 1 python driver_train_rom.py --intr_ld 10 --intf_ld 5 --nx 480 --ny 24 --act_type Swish
# jsrun -r 1 python driver_train_port_autoencoder.py --port_list 1 2 6 7 --port_ld 1 --port_row_nnz 3 --port_row_shift 3 --nx 480 --ny 24 --act_type Sigmoid
# jsrun -r 1 python driver_train_port_autoencoder.py --port_list 0 3 4 5 --port_ld 4 --port_row_nnz 3 --port_row_shift 3 --nx 480 --ny 24 --act_type Sigmoid
# jsrun -r 1 python driver_train_port_autoencoder.py --port_list 0 3 4 5 --port_ld 6 --port_row_nnz 3 --port_row_shift 3 --nx 480 --ny 24 --act_type Sigmoid
# jsrun -r 1 python driver_train_port_autoencoder.py --port_list 0 3 4 5 --port_ld 8 --port_row_nnz 3 --port_row_shift 3 --nx 480 --ny 24 --act_type Sigmoid