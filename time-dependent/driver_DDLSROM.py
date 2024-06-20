import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from utils.Burgers2D_probgen import Burgers2D
from utils.domain_decomposition import DD_FOM
from utils.lsrom import compute_bases_from_svd, DD_LSROM
import sys, os
import argparse
import dill as pickle

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
parser.add_argument("--intr_size", default=-1, type=int, 
                   help="Interior state POD basis size. -1 uses energy criterion.")
parser.add_argument("--intf_size", default=-1, type=int, 
                   help="Interface state POD basis size. -1 uses energy criterion.")
parser.add_argument("--port_size", default=-1, type=int, 
                   help="Port POD basis size. -1 uses energy criterion.")
parser.add_argument("--res_size", default=-1, type=int, 
                   help="Interior state POD basis size. -1 uses energy criterion.")
parser.add_argument("--ec_intr", default=1e-8, type=float, 
                    help="Energy criterion for interior state POD basis")
parser.add_argument("--ec_intf", default=1e-8, type=float, 
                    help="Energy criterion for interface state POD basis")
parser.add_argument("--ec_res", default=1e-8, type=float, 
                    help="Energy criterion for residual POD basis")
parser.add_argument("--ec_port", default=1e-8, type=float, 
                    help="Energy criterion for port POD basis")
parser.add_argument("--n_hrnodes", default=-1, type=int, 
                   help="Number of nodes for HR per subdomain. -1 uses sampling ratio.")
parser.add_argument("--hr_ratio", default=2.0, type=float, 
                   help="Ratio of HR nodes to residual basis size.")
parser.add_argument("--n_constraints", default=1, type=int, 
                   help="Number of weak constraints.")
parser.add_argument("--no_hr", action='store_true',
                   help="Pass to deactivate hyper reduction")
parser.add_argument("--full_subdomain", action='store_true', 
                   help="Pass to use full subdomain snapshot bases.")
parser.add_argument("--seed", default=0, type=int, 
                   help="Random seed")
parser.add_argument("--con_type", default='srpc', type=str, 
                   help="Constraint type. 'srpc' uses strong ROM-port constraints, 'wfpc' uses weak FOM-port constraints.")
parser.add_argument("--scaling", default=1, type=int, 
                   help="Residual scaling factor. Default is 1. -1 uses hx*hy")
args = parser.parse_args()

# parameters for physical domain and FD discretization
x_lim = [0, 1]
y_lim = [0, 1]
t_lim = [0, 2]
nx, ny, nt = args.nx, args.ny, args.nt
viscosity = args.viscosity
nsub_x, nsub_y = args.nsub_x, args.nsub_y
mu = args.mu

# energy criteria for POD bases
ec_res  = args.ec_res
ec_intr = args.ec_intr
ec_intf = args.ec_intf
ec_port = args.ec_port

# POD basis size. set to -1 to use energy criterion
nbasis_res  = args.res_size
nbasis_intr = args.intr_size
nbasis_intf = args.intf_size
nbasis_port = args.port_size

# LS-ROM and HR parameters
hr            = not args.no_hr
con_type      = args.con_type
n_constraints = args.n_constraints
sample_ratio  = args.hr_ratio
n_samples     = args.n_hrnodes
seed          = args.seed
scaling       = args.scaling

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
print(f'ec_res          = {ec_res}')
print(f'ec_intr         = {ec_intr}')
print(f'nbasis_res      = {nbasis_res:4d}    (-1 uses energy criterion)')
print(f'nbasis_intr     = {nbasis_intr:4d}    (-1 uses energy criterion)')
print(f'Use HR          = {hr}')
print(f'# HR nodes      = {n_samples:4d}    (-1 uses sample ratio)')
print(f'HR sample ratio = {sample_ratio}')
print(f'full sub basis  = {args.full_subdomain}')
print(f'scaling factor  = {scaling} (-1 uses hx*hy)')

print('\nConstraint parameters:')
print(f'Constraint type = {con_type}')
if con_type == 'srpc':
    con_dir = 'srpc/'
    print(f'ec_port         = {ec_port}')
    print(f'nbasis_port     = {nbasis_port:4d}    (-1 uses energy criterion)')
else:
    con_dir = f'wfpc_ncon_{n_constraints}/'
    print(f'ec_intf         = {ec_intf}')
    print(f'nbasis_intf     = {nbasis_intf:4d}    (-1 uses energy criterion)')
    print(f'n_constraints   = {n_constraints}')

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
    
# compute bases
print('\nComputing bases:')
sys.stdout.flush()
res_bases, intr_bases, intf_bases, port_bases = [], [], [], []
fs_str = 'full_subdomain_' if args.full_subdomain else ''

# residual and interior state bases
for i in range(ddfom.n_sub):
    sub_dir  = data_dir2 + f'sub_{i+1}of{ddfom.n_sub}/'
    intr_dir = sub_dir + 'interior/'
      
    res_dict  = pickle.load(open(sub_dir  + f'residual_svd_data.p', 'rb'))
    intr_dict = pickle.load(open(intr_dir + f'{fs_str}svd_data.p', 'rb'))
    
    res_bases.append(compute_bases_from_svd(res_dict, ec=ec_res, nbasis=nbasis_res))
    intr_bases.append(compute_bases_from_svd(intr_dict, ec=ec_intr, nbasis=nbasis_intr))
    
    print(f'\nSubdomain {i}:')
    print(f'residual_basis.shape  = {res_bases[i].shape}')
    print(f'interior_basis.shape  = {intr_bases[i].shape}')
    sys.stdout.flush()

# interface state bases
print('')
if con_type == 'srpc':
    n_ports = len(ddfom.ports)
    for j in range(len(ddfom.ports)):
        port_file = data_dir2 + f'port_{j+1}of{n_ports}_svd_data.p'
        port_dict  = pickle.load(open(port_file, 'rb'))
        port_bases.append(compute_bases_from_svd(port_dict, ec=ec_port, nbasis=nbasis_port)) 
        print(f'port_basis[{j}].shape  = {port_bases[j].shape}')

else:
    for i in range(ddfom.n_sub):
        sub_dir  = data_dir2 + f'sub_{i+1}of{ddfom.n_sub}/'
        intf_dir = sub_dir + 'interface/'

        intf_dict = pickle.load(open(intf_dir + f'{fs_str}svd_data.p', 'rb'))

        intf_bases.append(compute_bases_from_svd(intf_dict, ec=ec_intf, nbasis=nbasis_intf)) 

        print(f'Subdomain {i}:')
        print(f'interface_basis.shape = {intf_bases[i].shape}\n')        
        sys.stdout.flush()
      
print('\nInitializing DD LS-ROM...')
sys.stdout.flush()
ddlsrom = DD_LSROM(ddfom, 
                   intr_bases,
                   intf_bases,
                   res_bases=res_bases,
                   port_bases=port_bases,
                   constraint_type=con_type,
                   hr=hr,
                   sample_ratio=sample_ratio,
                   n_samples=n_samples,
                   n_corners=-1,
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
rom_figs0 = fig_dir2 + f'ddlsrom_{nsub_x}x_by_{nsub_y}y/'
rom_figs  = rom_figs0 + con_dir
for d in [fig_dir2, rom_figs0, rom_figs]:
    if not os.path.exists(d): os.mkdir(d)

# set initial condition
ddlsrom.set_initial(lambda xy: u0(xy, mu), lambda xy: v0(xy, mu))

# solve Burgers equation
print('\nSolve DD LS-ROM:') if not hr else print('\nSolve DD LS-ROM-HR:')
sys.stdout.flush()
w_intr, w_intf, u_intr, v_intr, u_intf, v_intf, u_full, v_full, lam, romtime, ih, flag = ddlsrom.solve(t_lim, 
                                                                                             nt, 
                                                                                             tol=args.tol, 
                                                                                             maxit=args.maxit)
print('Done!') 
sys.stdout.flush()

# store snapshots
print('\nSaving data...')
sys.stdout.flush()
intr_str = f'ecintr_{ec_intr:1.0e}_' if nbasis_intr < 0 else f'intr_{nbasis_intr}_'
if con_type == 'srpc':
    intf_str = f'ecport_{ec_port:1.0e}_' if nbasis_port < 0 else f'port_{nbasis_port}_'
else:
    intf_str = f'ecintf_{ec_intf:1.0e}_' if nbasis_intf < 0 else f'intf_{nbasis_intf}_'
if hr:
    hr_str = f'hr_{n_samples}_' if n_samples > 0 else f'hr_{sample_ratio}x_'
else:
    hr_str = ''

# save SQP iteration history 
ih_filename = data_dir2 + f'mu_{mu}_{intr_str}{intf_str}{hr_str}ithist.p'
pickle.dump(ih, open(ih_filename, 'wb'))
print(f'Avg # of SQP iterations per time step = {np.mean(ih)}')

for i, s in enumerate(ddlsrom.subdomain):
    sub_dir  = data_dir2 + f'sub_{i+1}of{ddlsrom.n_sub}/'
    intr_dir = sub_dir + 'interior/' + con_dir
    intf_dir = sub_dir + 'interface/' + con_dir
    for d in [intr_dir, intf_dir]:
        if not os.path.exists(d): os.mkdir(d)

    intr_dict = {'solution': w_intr[i],
                 'basis':    s.interior.basis,
                 'runtime':  romtime}
    intf_dict = {'solution': w_intf[i],
                 'basis':    s.interface.basis,
                 'runtime':  romtime}

    intr_filename = intr_dir + f'lsrom_{intr_str}{hr_str}mu_{mu}_state.p'
    intf_filename = intf_dir + f'lsrom_{intf_str}{hr_str}mu_{mu}_state.p'

    pickle.dump(intr_dict, open(intr_filename, 'wb'))
    pickle.dump(intf_dict, open(intf_filename, 'wb'))
print('Done!')

ntr    = u_full.shape[0]
abserr = np.zeros(ntr)
for i, s in enumerate(ddlsrom.subdomain):
    abserr += np.sum(np.square(ufom_intr[i][:ntr]-u_intr[i]), 1) + \
              np.sum(np.square(vfom_intr[i][:ntr]-v_intr[i]), 1) + \
              np.sum(np.square(ufom_intf[i][:ntr]-u_intf[i]), 1) + \
              np.sum(np.square(vfom_intf[i][:ntr]-v_intf[i]), 1)

abserr = np.max(np.sqrt(ddfom.hx*ddfom.hy*abserr/ddfom.n_sub))
HR_str = '-HR' if hr else ''
print(f'\nDD LS-ROM{HR_str} absolute error = {abserr:1.4e}\n')

print(f'FOM runtime = {fomtime:1.5e} seconds')
print(f'ROM runtime = {romtime:1.5e} seconds')
print(f'Speedup     = {fomtime/romtime}')
sys.stdout.flush()

# save gifs of solutions
umin = np.min([u.min() for u in ufom_intr + ufom_intf])
umax = np.max([u.max() for u in ufom_intr + ufom_intf])
vmin = np.min([v.min() for v in vfom_intr + vfom_intf])
vmax = np.max([v.max() for v in vfom_intr + vfom_intf])

prefix = rom_figs + f'{intr_str}{intf_str}{fs_str}{hr_str}'

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
