#BSUB -nnodes 1
#BSUB -W 01:00
#BSUB -q pbatch
#BSUB -J test_2x1
#BSUB -o job_outputs/testing/nmrom_visc1e-4/test_2x1.txt

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

# solution parameters
mu=1.0          # amplitude of initial condition
maxit=10        # maximum number of iterations for Lagrange-Gauss-Newton SQP
tol=1e-5        # stopping tolerance for Lagrange-Gauss-Newton SQP

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

ec_res=1e-16

batch=32
loss='RelMSE'   # AbsMSE or RelMSE

con_type='srpc' # constraint type. 'srpc' (strong rom-port constraints) or 'wfpc' (weak fom-port constraints)
ncon=1          # number of weak constraints

no_hr=''
hr_ratio=2.0    # ratio of HR nodes to size of residual basis
n_hrnodes=-1   # number of HR nodes (-1 uses sample ratio)
n_corners=-1   # number of corner (interface) HR nodes (-1 uses 0.75*n_hrnodes)
seed=0          # random seed

rbf='--rbf'     # use rbf initial guess
rbf_snaps=100   # number of training snapshots for rbf interpolants
scaling=-1      # residual scaling factor. -1 uses hx*hy

echo $HOSTNAME
# compute solution for DD NM-ROM
jsrun -r 1 python driver_DDNMROM.py --nx $nx --ny $ny --nt $nt --viscosity $visc --mu $mu --nsub_x $nsub_x --nsub_y $nsub_y --maxit $maxit --tol $tol --intr_size $intr_size --intf_size $intf_size --port_size $port_size --intr_row_nnz $intr_row_nnz --intf_row_nnz $intf_row_nnz --port_row_nnz $port_row_nnz --intr_row_shift $intr_row_shift --intf_row_shift $intf_row_shift --port_row_shift $port_row_shift --ec_res $ec_res --hr_ratio $hr_ratio --n_constraints $ncon --n_hrnodes $n_hrnodes --seed $seed --con_type $con_type --intr_act $intr_act --intf_act $intf_act --port_act $port_act --batch $batch --loss $loss $no_hr --n_corners $n_corners --wd_intr $wd_intr --wd_intf $wd_intf --wd_port $wd_port --rbf $rbf --scaling $scaling --rbf_snaps $rbf_snaps
