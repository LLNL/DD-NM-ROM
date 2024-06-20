#BSUB -nnodes 1
#BSUB -W 12:00
#BSUB -q pbatch
#BSUB -J nsnaps
#BSUB -o nsnaps_fine_ports.txt

# python/pytorch/horovod is forking a process,
# this setting is needed so that MPI will still work after a fork()
export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

echo $HOSTNAME

# train DD-NM-ROMs for different number of snapshot training data
jsrun -r 1 python driver_train_port_autoencoder.py --nx 480 --ny 24 --act_type Sigmoid --nsnaps 1600
jsrun -r 1 python driver_train_port_autoencoder.py --nx 480 --ny 24 --act_type Sigmoid --nsnaps 3200
jsrun -r 1 python driver_train_port_autoencoder.py --nx 480 --ny 24 --act_type Sigmoid --nsnaps 4800
jsrun -r 1 python driver_train_rom.py --nsnaps 1600 --intr_ld 8 --intr_only --nx 480 --ny 24 --act_type Swish
jsrun -r 1 python driver_train_rom.py --nsnaps 3200 --intr_ld 8 --intr_only --nx 480 --ny 24 --act_type Swish
jsrun -r 1 python driver_train_rom.py --nsnaps 4800 --intr_ld 8 --intr_only --nx 480 --ny 24 --act_type Swish