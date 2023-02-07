#BSUB -nnodes 1
#BSUB -W 12:00
#BSUB -q pbatch
#BSUB -J train_rom
#BSUB -o training.txt

# python/pytorch/horovod is forking a process,
# this setting is needed so that MPI will still work after a fork()
export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

echo $HOSTNAME

jsrun -r 1 python driver_train_rom.py --nx 480 --ny 24 --intr_ld 16 --intf_ld 8 --act_type Swish