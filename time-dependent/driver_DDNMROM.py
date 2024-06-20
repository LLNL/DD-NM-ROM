import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from utils.Burgers2D_probgen import Burgers2D
from utils.domain_decomposition import DD_FOM
from utils.nmrom import DD_NMROM, dd_rbf
from utils.lsrom import compute_bases_from_svd, assemble_snapshot_matrix
import sys, os
import argparse
import dill as pickle
import torch

# make directories for figures and data
data_dir = './data/'
fig_dir0 = './figures/'
for d in [data_dir, fig_dir0]:
    if not os.path.exists(d): os.mkdir(d)
        
parser = argparse.ArgumentParser()
parser.add_argument("--nsub_x", default=2, type=int, 
                   help="number of subdomains in x-direction")
parser.add_argument("--nsub_y", default=2, type=int, 
                   help="number of subdomains in y-direction")

parser.add_argument("--nx", default=60, type=int, 
                   help="number of grid points in x-direction")
parser.add_argument("--ny", default=60, type=int, 
                   help="number of grid points in y-direction")
parser.add_argument("--nt", default=1500, type=int, 
                   help="number of time steps")
parser.add_argument("--viscosity", default=1e-4, type=float, 
                    help="viscosity for 2D Burgers' problem")

parser.add_argument("--maxit", default=20, type=int, 
                    help="maximum number of iterations for Newton solver")
parser.add_argument("--tol", default=1e-4, type=float, 
                    help="absolute tolerance for Newton solver")

parser.add_argument("--mu", default=1.0, type=float, 
                    help="Amplitude of initial condition (parameter for PDE)")

parser.add_argument("--intr_size", default=5, type=int, 
                   help="ROM dimension for interior states")
parser.add_argument("--intf_size", default=5, type=int, 
                   help="ROM dimension for interface states")
parser.add_argument("--port_size", default=2, type=int, 
                   help="ROM dimension for port states")

parser.add_argument("--intr_row_nnz", default=5, type=int, 
                   help="Number of nonzeros per row (col) in interior decoder (encoder) mask")
parser.add_argument("--intf_row_nnz", default=5, type=int, 
                   help="Row (col) shift in interface decoder (encoder) mask")
parser.add_argument("--port_row_nnz", default=2, type=int, 
                   help="Row (col) shift in port decoder (encoder) mask")

parser.add_argument("--intr_row_shift", default=5, type=int, 
                   help="Row (col) shift in interior decoder (encoder) mask")
parser.add_argument("--intf_row_shift", default=5, type=int, 
                   help="Number of nonzeros per row (col) in interface decoder (encoder) mask")
parser.add_argument("--port_row_shift", default=2, type=int, 
                   help="Row (col) shift in port decoder (encoder) mask")

parser.add_argument("--res_size", default=-1, type=int, 
                   help="Interior state POD basis size. -1 uses energy criterion.")
parser.add_argument("--ec_res", default=1e-8, type=float, 
                    help="Energy criterion for residual POD basis")

parser.add_argument("--n_hrnodes", default=-1, type=int, 
                   help="Number of nodes for HR per subdomain. -1 uses sampling ratio.")
parser.add_argument("--n_corners", default=-1, type=int, 
                   help="Number of corner (interface) nodes for HR per subdomain. -1 0.75*n_hrnodes.")
parser.add_argument("--hr_ratio", default=2.0, type=float, 
                   help="Ratio of HR nodes to residual basis size.")
parser.add_argument("--n_constraints", default=1, type=int, 
                   help="Number of weak constraints.")
parser.add_argument("--no_hr", action='store_true',
                   help="Pass to deactivate hyper reduction")
parser.add_argument("--rbf", action='store_true',
                   help="Pass to use RBF initial guess")

parser.add_argument("--intr_act", default = 'Swish', type=str,
                   help="Interior state activation type. Only Sigmoid and Swish are implemented")
parser.add_argument("--intf_act", default = 'Swish', type=str,
                   help="Interface state activation type. Only Sigmoid and Swish are implemented")
parser.add_argument("--port_act", default = 'Swish', type=str,
                   help="Interface state activation type. Only Sigmoid and Swish are implemented")

parser.add_argument("--batch", default=32, type=int, 
                   help="Batch size")
parser.add_argument("--loss", default='AbsMSE', type=str,
                    help="loss type. 'AbsMSE' or 'RelMSE.'")

parser.add_argument("--wd_intr", default=0, type=float, 
                   help="Weight decay for interior AEs, i.e. L2 regularization parameter.")
parser.add_argument("--wd_intf", default=0, type=float, 
                   help="Weight decay for interface AEs, i.e. L2 regularization parameter.")
parser.add_argument("--wd_port", default=0, type=float, 
                   help="Weight decay for interface AEs, i.e. L2 regularization parameter.")

parser.add_argument("--seed", default=0, type=int, 
                   help="Random seed")
parser.add_argument("--con_type", default='srpc', type=str, 
                   help="Constraint type. 'srpc' uses strong ROM-port constraints, 'wfpc' uses weak FOM-port constraints.")

parser.add_argument("--scaling", default=1, type=int, 
                   help="Residual scaling factor. Default is 1. -1 uses hx*hy")
parser.add_argument("--rbf_snaps", default=400, type=int, 
                   help="Number of snapshots used for training RBF interpolant. Default is 400.")
args = parser.parse_args()

# parameters for physical domain and FD discretization
x_lim = [0, 1]
y_lim = [0, 1]
t_lim = [0, 2]
nx, ny, nt = args.nx, args.ny, args.nt
viscosity = args.viscosity
nsub_x, nsub_y = args.nsub_x, args.nsub_y
n_sub = nsub_x*nsub_y
mu = args.mu

# size of latent dimension
intr_size, intf_size, port_size = args.intr_size, args.intf_size, args.port_size
ec_res = args.ec_res
nbasis_res = args.res_size

# mask parameters
intr_rnnz = args.intr_row_nnz
intf_rnnz = args.intf_row_nnz
port_rnnz = args.port_row_nnz
intr_rshift = args.intr_row_shift
intf_rshift = args.intf_row_shift
port_rshift = args.port_row_shift
batch       = args.batch
loss        = args.loss
wd_intr     = args.wd_intr
wd_intf     = args.wd_intf
wd_port     = args.wd_port


# NM-ROM and HR parameters
hr            = not args.no_hr
rbf           = args.rbf
con_type      = args.con_type
n_constraints = args.n_constraints
sample_ratio  = args.hr_ratio
n_samples     = args.n_hrnodes
n_corners     = args.n_corners
seed          = args.seed
scaling       = args.scaling
rbf_snaps     = args.rbf_snaps

print('\nDD FOM parameters:')
print(f'nsub_x    = {nsub_x}')
print(f'nsub_y    = {nsub_y}')
print(f'nx        = {nx}')
print(f'ny        = {ny}')
print(f'nt        = {nt}')
print(f'viscosity = {viscosity}')
print(f'mu        = {mu}')
print(f'seed      = {seed}')

print('\nROM parameters:')
print(f'Use HR          = {hr}')
print(f'ec_res          = {ec_res}')
print(f'# HR nodes      = {n_samples:4d}    (-1 uses sample ratio)')
print(f'# corner nodes  = {n_corners:4d}    (-1 uses 0.75*hr_nodes)')
print(f'HR sample ratio = {sample_ratio}')
print(f'scaling factor  = {scaling} (-1 uses hx*hy)\n')
print(f'#rbf train data = {rbf_snaps}')

print(f'interior ROM size   = {intr_size}')
print(f'interior row_nnz    = {intr_rnnz}')
print(f'interior row_shift  = {intr_rshift}')
print(f'interior activation = {args.intr_act}')

print('\nConstraint parameters:')
print(f'Constraint type = {con_type}')
if con_type == 'srpc':
    con_dir = 'srpc/'
    print(f'port ROM size   = {port_size}')
    print(f'port row_nnz    = {port_rnnz}')
    print(f'port row_shift  = {port_rshift}')
    print(f'port activation = {args.port_act}')

else:
    con_dir = f'wfpc_ncon_{n_constraints}/'
    print(f'interface ROM size   = {intf_size}')
    print(f'interface row_nnz    = {intf_rnnz}')
    print(f'interface row_shift  = {intf_rshift}')
    print(f'n_constraints        = {n_constraints}')
    print(f'interface activation = {args.intf_act}')

print(f'\nbatch size   = {batch}')
print(f'loss type    = {loss}')
print(f'intr wd      = {wd_intr}')
print(f'intf wd      = {wd_intf}')
print(f'port wd      = {wd_port}\n')

print('\nParameters for Newton solver:')
print(f'maxit = {args.maxit}')
print(f'tol   = {args.tol}')
sys.stdout.flush()

# initialize monolithic model
print('\nInitializing monolithic Burgers model...')
sys.stdout.flush()
fom = Burgers2D(nx, ny, x_lim, y_lim, viscosity)
print('Done!')

# initialize DD FOM
print('\nInitializing DD FOM...')
sys.stdout.flush()
ddfom = DD_FOM(fom, nsub_x, nsub_y)
print('Done!')
sys.stdout.flush()

# parameterized initial conditions
def u0(XY, mu):
    val = np.zeros(len(XY))
    for i, xy in enumerate(XY):
        if np.all([xy[0] >= 0.0, xy[0] <= 0.5, xy[1] >= 0.0, xy[1] <= 0.5]):
            val[i] = mu*np.sin(2*np.pi*xy[0])*np.sin(2*np.pi*xy[1])
    return val 

def v0(XY, mu):
    val = np.zeros(len(XY))
    for i, xy in enumerate(XY):
        if np.all([xy[0] >= 0.0, xy[0] <= 0.5, xy[1] >= 0.0, xy[1] <= 0.5]):
            val[i] = mu*np.sin(2*np.pi*xy[0])*np.sin(2*np.pi*xy[1])
    return val 
    
# make figure directory
fig_dir1  = fig_dir0 + f'nx_{nx}_ny_{ny}_nt_{nt}_visc_{viscosity}/'
data_dir1 = data_dir + f'nx_{nx}_ny_{ny}_nt_{nt}_visc_{viscosity}/'
data_dir2 = data_dir1 + f'DD_{nsub_x}x_by_{nsub_y}y/'
for d in [fig_dir1, data_dir1, data_dir2]:
    if not os.path.exists(d): os.mkdir(d)
        
# frame updater for animation
X, Y = np.meshgrid(fom.xx, fom.yy)
ht   = (t_lim[1]-t_lim[0])/nt
def update_frame(i, Z, zmin, zmax, cb_label):
    plt.clf()
    plt.pcolormesh(X, Y, Z[i], cmap='viridis', shading='auto', vmin=zmin, vmax=zmax) 
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('$t_{'+f'{i:4d}'+'}' + f'={ht*i+t_lim[0]:1.4f}$')
    cb = plt.colorbar(orientation='vertical', label=cb_label)
    return plt
    
# compute residual basis if HR is used
res_bases = []
if hr:
    print('\nComputing residual bases:')
    sys.stdout.flush()
    for i in range(ddfom.n_sub):
        sub_dir  = data_dir2 + f'sub_{i+1}of{ddfom.n_sub}/'
        res_dict  = pickle.load(open(sub_dir  + f'residual_svd_data.p', 'rb'))
        res_bases.append(compute_bases_from_svd(res_dict, ec=ec_res, nbasis=nbasis_res))

        print(f'Subdomain {i}:')
        print(f'residual_basis.shape  = {res_bases[i].shape}')
    sys.stdout.flush()

# load autoencoders
net_dir = f'./trained_nets/nx_{nx}_ny_{ny}_nt_{nt}_visc_{viscosity}/DD_{nsub_x}x_by_{nsub_y}y/'
intr_ae_list, intf_ae_list, port_ae_list = [], [], []

# construct file and directory names for loading trained nets
intr_file = f'ld_{intr_size}_rnnz_{intr_rnnz}_rshift_{intr_rshift}_'  + \
            f'{args.intr_act}_batch_{batch}_{loss}loss_wd{wd_intr}.p'
print('\nLoading interior state autoencoders...')
sys.stdout.flush()
for i in range(ddfom.n_sub):
    intr_dir = net_dir+f'sub_{i+1}of{ddfom.n_sub}/interior/'
    intr_ae_list.append(torch.load(intr_dir+intr_file))
print('Done!\n')
sys.stdout.flush()

# interface state bases
print('')
if con_type == 'srpc':
    print('Loading port state autoencoders...')
    sys.stdout.flush()
    port_size_list = [min(port_size, 2*len(ddfom.port_dict[p])) for p in ddfom.ports]
    n_ports = len(ddfom.ports)
    for i in range(n_ports):
        port_dir  = net_dir + f'port_{i+1}of{n_ports}/'
        port_file = f'ld_{port_size_list[i]}_rnnz_{port_rnnz}_rshift_{port_rshift}_'  + \
                    f'{args.port_act}_batch_{batch}_{loss}loss_wd{wd_port}.p'
        port_ae_list.append(torch.load(port_dir+port_file))

else:
    print('Loading interface state autoencoders...')
    sys.stdout.flush()
    intf_file = f'ld_{intf_size}_rnnz_{intf_rnnz}_rshift_{intf_rshift}_'  + \
                f'{args.intf_act}_batch_{batch}_{loss}loss_wd{wd_intf}.p'
    for i in range(ddfom.n_sub):
        intf_dir = net_dir+f'sub_{i+1}of{ddfom.n_sub}/interface/'
        intf_ae_list.append(torch.load(intf_dir+intf_file))
        
print('Done!\n')
sys.stdout.flush()      

print('Initializing DD NM-ROM...')
sys.stdout.flush()
ddnmrom = DD_NMROM(ddfom, 
                   intr_ae_list, 
                   intf_ae_list=intf_ae_list,
                   port_ae_list=port_ae_list, 
                   res_bases=res_bases,
                   constraint_type=con_type, 
                   hr=hr,
                   sample_ratio=sample_ratio,
                   n_samples=n_samples,
                   n_corners=n_corners,
                   n_constraints=n_constraints, 
                   seed=seed, 
                   scaling=scaling)
print('Done!')

print(f'\nRunning for mu = {mu}:')
sys.stdout.flush()

# load DD FOM data
print(f'\nLoading FOM data...')
sys.stdout.flush()

ufom_intr, vfom_intr, ufom_intf, vfom_intf = [], [], [], []
for i, s in enumerate(ddfom.subdomain):
    sub_dir = data_dir2 + f'sub_{i+1}of{ddfom.n_sub}/'
    uv_intr = pickle.load(open(sub_dir + f'interior/mu_{mu}_uv_state.p', 'rb'))
    fomtime = uv_intr['runtime']
    uv_intr = uv_intr['solution']
    ufom_intr.append(uv_intr[:, :s.interior.indices.size])
    vfom_intr.append(uv_intr[:, s.interior.indices.size:])

    uv_intf = pickle.load(open(sub_dir + f'interface/mu_{mu}_uv_state.p', 'rb'))
    fomtime = uv_intf['runtime']
    uv_intf = uv_intf['solution']
    ufom_intf.append(uv_intf[:, :s.interface.indices.size])
    vfom_intf.append(uv_intf[:, s.interface.indices.size:])
print(f'Done!')
sys.stdout.flush()


# make figure directories
fig_dir2  = fig_dir1 + f'mu_{mu}/' 
rom_figs0 = fig_dir2 + f'ddnmrom_{nsub_x}x_by_{nsub_y}y/'
rom_figs  = rom_figs0 + con_dir
for d in [fig_dir2, rom_figs0, rom_figs]:
    if not os.path.exists(d): os.mkdir(d)

# set initial condition
ddnmrom.set_initial(lambda xy: u0(xy, mu), lambda xy: v0(xy, mu))

if rbf:
    # train RBF model
    print('Training RBF interpolant...')
    sys.stdout.flush()
    mu_list   = [0.9, 0.95, 1.05, 1.1]
    snapshots = assemble_snapshot_matrix(data_dir1, mu_list)
    rbfmdl    = dd_rbf(ddnmrom, t_lim, nt, mu_list, snapshots, n_snaps=rbf_snaps)
    print('Done!\n')
    sys.stdout.flush()

    # computing initial guess
    print('Computing initial guess...')
    sys.stdout.flush()
    guess, rbftime = rbfmdl.compute_guess(t_lim, nt, mu, ddnmrom)
    print('Done!\n')
    sys.stdout.flush()
else: 
    guess = []
    
# solve Burgers equation
print('\nSolve DD NM-ROM:') if not hr else print('\nSolve DD NM-ROM-HR:')
sys.stdout.flush()
w_intr, w_intf, u_intr, v_intr, u_intf, v_intf, u_full, v_full, lam, romtime, ih, flag = ddnmrom.solve(t_lim, 
                                                                                                   nt, 
                                                                                                   guess=guess,
                                                                                                   tol=args.tol, 
                                                                                                   maxit=args.maxit)
if rbf: romtime+= rbftime

print('Done!') 
sys.stdout.flush()

# store snapshots
print('\nSaving data...')
sys.stdout.flush()
intr_str = f'intr_ld{intr_size}rnz{intr_rnnz}rs{intr_rshift}wd{wd_intr}{args.intr_act}_'
if con_type == 'srpc':
    intf_str = f'port_ld{port_size}rnz{port_rnnz}rs{port_rshift}wd{wd_port}{args.port_act}_'
else:
    intf_str = f'intf_ld{intf_size}rnz{intf_rnnz}rs{intf_rshift}wd{wd_intf}{args.intf_act}_'
if hr:
    hr_str = f'hr_{n_samples}_' if n_samples > 0 else f'hr_{sample_ratio}x_'
else:
    hr_str = ''
    
# save SQP iteration history 
ih_filename = data_dir2 + f'mu_{mu}_{intr_str}{intf_str}{hr_str}ithist.p'
pickle.dump(ih, open(ih_filename, 'wb'))
print(f'Avg # of SQP iterations per time step = {np.mean(ih)}')

for i, s in enumerate(ddnmrom.subdomain):
    sub_dir  = data_dir2 + f'sub_{i+1}of{ddnmrom.n_sub}/'
    intr_dir = sub_dir + 'interior/' + con_dir
    intf_dir = sub_dir + 'interface/' + con_dir
    for d in [intr_dir, intf_dir]:
        if not os.path.exists(d): os.mkdir(d)

    intr_dict = {'solution': w_intr[i],
                 'runtime':  romtime}
    intf_dict = {'solution': w_intf[i],
                 'runtime':  romtime}

    intr_filename = intr_dir + f'nmrom_{intr_str}{hr_str}mu_{mu}_state.p'
    intf_filename = intf_dir + f'nmrom_{intf_str}{hr_str}mu_{mu}_state.p'

    pickle.dump(intr_dict, open(intr_filename, 'wb'))
    pickle.dump(intf_dict, open(intf_filename, 'wb'))
print('Done!')

ntr    = u_full.shape[0]
abserr = np.zeros(ntr)
for i, s in enumerate(ddnmrom.subdomain):
    abserr += np.sum(np.square(ufom_intr[i][:ntr]-u_intr[i]), 1) + \
              np.sum(np.square(vfom_intr[i][:ntr]-v_intr[i]), 1) + \
              np.sum(np.square(ufom_intf[i][:ntr]-u_intf[i]), 1) + \
              np.sum(np.square(vfom_intf[i][:ntr]-v_intf[i]), 1)

abserr = np.max(np.sqrt(ddfom.hx*ddfom.hy*abserr/ddfom.n_sub))
HR_str = '-HR' if hr else ''
print(f'\nDD NM-ROM{HR_str} absolute error = {abserr:1.4e}\n')

print(f'FOM runtime = {fomtime:1.5e} seconds')
print(f'ROM runtime = {romtime:1.5e} seconds')
print(f'Speedup     = {fomtime/romtime}')
sys.stdout.flush()

# save gifs of solutions
umin = np.min([u.min() for u in ufom_intr + ufom_intf])
umax = np.max([u.max() for u in ufom_intr + ufom_intf])
vmin = np.min([v.min() for v in vfom_intr + vfom_intf])
vmax = np.max([v.max() for v in vfom_intr + vfom_intf])

prefix = rom_figs + f'{intr_str}{intf_str}{hr_str}'

plt.rc('font', size=20)
plt.rcParams['text.usetex'] = True

print('\nGenerating gif for u state...')
sys.stdout.flush()
UU = u_full.reshape(ntr, ddfom.ny, ddfom.nx)
fig = plt.figure()
ani = animation.FuncAnimation(fig, lambda i: update_frame(i, UU, umin, umax, '$u(t, x,y)$'), frames=ntr, interval=1)
ani.save(prefix + 'u_state.gif', writer='imagemagick', fps=nt//10)
plt.close()
print('Done!')
print('u state gif = ' + prefix + 'u_state.gif')
sys.stdout.flush()

print('\nGenerating gif for v state...')
sys.stdout.flush()
VV = v_full.reshape(ntr, ddfom.ny, ddfom.nx)
fig = plt.figure()
ani = animation.FuncAnimation(fig, lambda i: update_frame(i, VV, vmin, vmax, '$v(t, x,y)$'), frames=ntr, interval=1)
ani.save(prefix + 'v_state.gif', writer='imagemagick', fps=nt//10)
plt.close()
print('Done!')
print('v state gif = ' + prefix + 'v_state.gif')
print('End')
sys.stdout.flush()
