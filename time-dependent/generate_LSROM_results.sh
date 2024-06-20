#BSUB -nnodes 1
#BSUB -W 01:00
#BSUB -q pbatch
#BSUB -J lsrom2x1_visc1e-4_hr_86
#BSUB -o job_outputs/testing/lsrom_visc1e-4/lsrom2x1_visc1e-4_hr_86.txt

export IBV_FORK_SAFE=1
export OMP_NUM_THREADS=1

# FOM parameters
nx=60           # spatial discetization in x direction
ny=60           # spatial discetization in x direction
nt=1500          # number of time steps
visc=1e-4       # viscosity

# DD parameters 
nsub_x=2        # number of subdomains in x direction
nsub_y=1        # number of subdomains in y direction

# solution parameters
mu=1.0          # amplitude of initial condition
maxit=12        # maximum number of iterations for Lagrange-Gauss-Newton SQP
tol=1e-9        # stopping tolerance for Lagrange-Gauss-Newton SQP

# ROM parameters
ec_res=1e-16    # energy criterion for residual POD basis
ec_intr=1e-8    # energy criterion for residual POD basis
ec_intf=1e-8    # energy criterion for residual POD basis
ec_port=1e-8    # energy criterion for port POD basis

intr_size=8     # number of DOF for interior state basis
intf_size=5     # number of DOF for interface state basis
port_size=6    # number of DOF for port basis

con_type='srpc' # constraint type. 'srpc' (strong rom-port constraints) or 'wfpc' (weak fom-port constraints)
ncon=1          # number of weak constraints

no_hr='' 
hr_ratio=2.0    # ratio of HR nodes to size of residual basis
n_hrnodes=-1    # number of HR nodes (-1 uses sample ratio)

seed=0          # random seed
scaling=-1       # residual scaling factor. -1 uses hx*hy

echo $HOSTNAME
# # compute snapshots using monolithic FOM
# jsrun -r 1 python driver_monolithic_FOM.py --nx $nx --ny $ny --nt $nt --viscosity $visc --mu_list 0.9 0.95 1.0 1.05 1.1

# compute SVD data for POD bases
# jsrun -r 1 python collect_DD_SVD_data.py --nx $nx --ny $ny --nt $nt --viscosity $visc --nsub_x $nsub_x --nsub_y $nsub_y --mu_list 0.9 0.95 1.05 1.1

# # compute solution of DD FOM
# jsrun -r 1 python driver_DDFOM.py --nx $nx --ny $ny --nt $nt --viscosity $visc --nsub_x $nsub_x --nsub_y $nsub_y --mu $mu

# compute solution for DD LS-ROM
jsrun -r 1 python driver_DDLSROM.py --nx $nx --ny $ny --nt $nt --viscosity $visc --mu $mu --nsub_x $nsub_x --nsub_y $nsub_y --maxit $maxit --tol $tol --intr_size $intr_size --intf_size $intf_size --ec_res $ec_res --hr_ratio $hr_ratio --n_constraints $ncon --n_hrnodes $n_hrnodes --seed $seed --con_type $con_type --port_size $port_size --ec_port $ec_port $no_hr --scaling $scaling
