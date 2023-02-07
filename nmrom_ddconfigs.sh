#BSUB -nnodes 1
#BSUB -W 12:00
#BSUB -q pbatch
#BSUB -J configs
#BSUB -o configs_fine_swish.txt

# python/pytorch/horovod is forking a process,
# this setting is needed so that MPI will still work after a fork()
export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

echo $HOSTNAME

# train DD-NM-ROMs for different DD configurations
jsrun -r 1 python driver_train_rom.py --n_sub_x 2 --n_sub_y 1 --intr_ld 8 --intr_only --nx 480 --ny 24 --act_type Swish
jsrun -r 1 python driver_train_rom.py --n_sub_x 4 --n_sub_y 1 --intr_ld 8 --intr_only --nx 480 --ny 24 --act_type Swish
jsrun -r 1 python driver_train_rom.py --n_sub_x 4 --n_sub_y 2 --intr_ld 8 --intr_only --nx 480 --ny 24 --act_type Swish