#BSUB -nnodes 1
#BSUB -W 12:00
#BSUB -q pbatch
#BSUB -J romsize
#BSUB -o romsize_fine.txt

# python/pytorch/horovod is forking a process,
# this setting is needed so that MPI will still work after a fork()
export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

echo $HOSTNAME

# train DD-NM-ROMs for different ROM sizes
jsrun -r 1 python driver_train_rom.py --intr_ld 4 --intf_ld 4 --nx 480 --ny 24 --act_type Swish
jsrun -r 1 python driver_train_rom.py --intr_ld 6 --intf_ld 8 --nx 480 --ny 24 --act_type Swish
jsrun -r 1 python driver_train_rom.py --intr_ld 10 --intf_ld 10 --nx 480 --ny 24 --act_type Swish
