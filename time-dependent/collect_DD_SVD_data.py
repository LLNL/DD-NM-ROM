import numpy as np
import scipy.sparse as sp
from utils.Burgers2D_probgen import Burgers2D
from utils.domain_decomposition import DD_FOM
from utils.lsrom import assemble_snapshot_matrix, save_svd_data, save_full_subdomain_svd
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
parser.add_argument("--mu_list", nargs='*', type=float,
                   help="List of samples at which to solve Burgers' equation.", default=[])
args = parser.parse_args()

# parameters at which to collect snapshot data
mu_list = args.mu_list #[0.9, 0.95, 1.0, 1.05, 1.1]

# parameters for physical domain and FD discretization
x_lim = [0, 1]
y_lim = [0, 1]
t_lim = [0, 2]
nx, ny, nt = args.nx, args.ny, args.nt
viscosity = args.viscosity
nsub_x, nsub_y = args.nsub_x, args.nsub_y

print('\nCollecting SVD data for DD configuration with following parameters:')
print(f'nsub_x    = {nsub_x}')
print(f'nsub_y    = {nsub_y}')
print(f'nx        = {nx}')
print(f'ny        = {ny}')
print(f'nt        = {nt}')
print(f'viscosity = {viscosity}')
print(f'mu_list   = {mu_list}')

# initialize monolithic model
print('\nInitializing monolithic Burgers model...')
sys.stdout.flush()
fom = Burgers2D(nx, ny, x_lim, y_lim, viscosity)
print('Done!')

# initialize DD model
print('\nInitializing DD model...')
sys.stdout.flush()
ddfom = DD_FOM(fom, nsub_x, nsub_y)
print('Done!')
sys.stdout.flush()
   
# make figure directory
fig_dir1  = fig_dir0 + f'nx_{nx}_ny_{ny}_nt_{nt}_visc_{viscosity}/'
data_dir1 = data_dir + f'nx_{nx}_ny_{ny}_nt_{nt}_visc_{viscosity}/'
data_dir2 = data_dir1 + f'DD_{nsub_x}x_by_{nsub_y}y/'
for d in [fig_dir1, data_dir1, data_dir2]:
    if not os.path.exists(d): os.mkdir(d)

# load snapshot data
data = assemble_snapshot_matrix(data_dir1, mu_list)

for i,s in enumerate(ddfom.subdomain):   
    print(f'\nComputing SVD data for subdomain {i+1} of {ddfom.n_sub}:')
    sys.stdout.flush()
    
    sub_dir  = data_dir2 + f'sub_{i+1}of{ddfom.n_sub}/'
    intr_dir = sub_dir + 'interior/'
    intf_dir = sub_dir + 'interface/'
    for d in [sub_dir, intr_dir, intf_dir]:
        if not os.path.exists(d): os.mkdir(d)
    
    print(f'\nInterior state data...')
    sys.stdout.flush()
    intr_data = np.vstack([data[s.interior.indices],data[s.interior.indices+ddfom.nxy]])
    save_svd_data(intr_data, intr_dir+'svd_data.p')
    print('Done!')
    sys.stdout.flush()
    
    print(f'\nInterface state data...')
    sys.stdout.flush()
    intf_data = np.vstack([data[s.interface.indices],data[s.interface.indices+ddfom.nxy]])
    save_svd_data(intf_data, intf_dir+'svd_data.p')
    print('Done!')
    sys.stdout.flush()
    
    print('\nFull subdomain state data...')
    sys.stdout.flush()
    save_full_subdomain_svd(intr_data, intf_data, intr_dir+'full_subdomain_svd_data.p', intf_dir+'full_subdomain_svd_data.p')
    print('Done!')
    sys.stdout.flush()
    
    print(f'\nResidual data...')
    sys.stdout.flush()
    res_data = np.hstack([sp.block_diag([s.interior.I, s.interior.I])@intr_data, 
                          sp.block_diag([s.interface.I, s.interface.I])@intf_data])
    save_svd_data(res_data, sub_dir+'residual_svd_data.p')
    print('Done!')
    sys.stdout.flush()
    
n_ports = len(ddfom.ports)
print(f'\nComputing SVD data for {n_ports} ports:\n')
sys.stdout.flush()
for j, p in enumerate(ddfom.ports):
    print(f'Port {j+1}...')
    sys.stdout.flush()
    
    port_data = np.vstack([data[ddfom.port_dict[p]], data[ddfom.port_dict[p]+ddfom.nxy]])
    port_file = data_dir2 + f'port_{j+1}of{n_ports}_svd_data.p'
    save_svd_data(port_data, port_file)
    print('Done!')
    sys.stdout.flush()
    