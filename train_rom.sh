#BSUB -nnodes 1
#BSUB -W 3:00
#BSUB -q pbatch
#BSUB -J dense10
#BSUB -o dense10.txt

# python/pytorch/horovod is forking a process,
# this setting is needed so that MPI will still work after a fork()
export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

echo $HOSTNAME

# jsrun -r 1 python driver_train_rom.py --nx 480 --ny 24 --intr_ld 6 --intf_ld 4 --act_type Swish --intr_only
jsrun -r 1 python driver_train_monolithic_rom.py --nx 240 --ny 12 --ld 4 --act_type Swish --dense_encoder --enc_hidden 3
# jsrun -r 1 python driver_train_monolithic_rom.py --nx 240 --ny 12 --ld 4 --act_type Swish