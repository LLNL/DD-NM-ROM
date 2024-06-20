#BSUB -nnodes 1
#BSUB -W 3:00
#BSUB -q pbatch
#BSUB -J train2x1_visc1e-4
#BSUB -o job_outputs/training/nmrom_visc1e-4/train2x1.txt

export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

# FOM parameters
nx=60           # spatial discetization in x direction
ny=60           # spatial discetization in x direction
nt=1500         # number of time steps
visc=1e-4       # viscosity

# DD parameters 
nsub_x=2        # number of subdomains in x direction
nsub_y=1        # number of subdomains in y direction

# NM-ROM parameters
intr_size=8
intr_row_nnz=5
intr_row_shift=5
intr_act='Swish'
wd_intr=1e-6

intf_size=6
intf_row_nnz=5
intf_row_shift=5
intf_act='Swish'
wd_intf=1e-6

port_size=6
port_row_nnz=5
port_row_shift=5
port_act='Swish'
wd_port=1e-6

comps="interior interface port" # specify "interior", "interface", and/or "port" for training
sub_list=''
port_list=''
loss='RelMSE'   # AbsMSE or RelMSE

# AE training parameters
epochs=2000
batch=32
esp=300
lr=1e-3
lrp=50

echo $HOSTNAME
# compute solution for DD LS-ROM
jsrun -r 1 python driver_train_NMROM.py --nx $nx --ny $ny --nt $nt --viscosity $visc --nsub_x $nsub_x --nsub_y $nsub_y --intr_size $intr_size --intf_size $intf_size --port_size $port_size --intr_row_nnz $intr_row_nnz --intf_row_nnz $intf_row_nnz --port_row_nnz $port_row_nnz --intr_row_shift $intr_row_shift --intf_row_shift $intf_row_shift --port_row_shift $port_row_shift --epochs $epochs --batch $batch --lr_patience $lrp --es_patience $esp --intr_act $intr_act --intf_act $intf_act --port_act $port_act --mu_list 0.9 0.95 1.05 1.1 --comps $comps --loss $loss --lr $lr --sub_list $sub_list --port_list $port_list --wd_intr $wd_intr --wd_intf $wd_intf --wd_port $wd_port
