import torch
from torch import nn
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import dill as pickle
from time import time
import sys, os, copy
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse

from utils.Burgers2D_probgen import Burgers2D
from utils.domain_decomposition import DD_model
from utils.NM_ROM import separate_snapshots
from utils.autoencoder import Autoencoder, Encoder, Decoder

# Parser for command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--nx", default=480, type=int, 
                   help="number of grid points in x-direction")
parser.add_argument("--ny", default=24, type=int, 
                   help="number of grid points in y-direction")
parser.add_argument("--n_sub_x", default=2, type=int, 
                  help="number of subdomains in x-direction")
parser.add_argument("--n_sub_y", default=2, type = int,
                  help="number of subdomains in y-direction")
parser.add_argument("--intr_ld", default=8, type=int, 
                   help="ROM dimension for interior states")
parser.add_argument("--intf_ld", default=4, type=int, 
                   help="ROM dimension for interface states")
parser.add_argument("--intr_row_nnz", default=5, type=int, 
                   help="Number of nonzeros per row (col) in interior decoder (encoder) mask")
parser.add_argument("--intf_row_nnz", default=5, type=int, 
                   help="Row (col) shift in interface decoder (encoder) mask")
parser.add_argument("--intr_row_shift", default=5, type=int, 
                   help="Row (col) shift in interior decoder (encoder) mask")
parser.add_argument("--intf_row_shift", default=5, type=int, 
                   help="Number of nonzeros per row (col) in interface decoder (encoder) mask")
parser.add_argument("--nsnaps", default=6400, type=int, 
                   help="Number of snapshots used for training")
parser.add_argument("--batch", default=32, type=int, 
                   help="batch size for training")
parser.add_argument("--intr_only", action='store_true',
                   help="Only train autoencoders for interior states")
parser.add_argument("--intf_only", action='store_true', 
                   help="Only train autoencoders for interface states")
parser.add_argument("--act_type", default = 'Swish',
                   help="Activation type. Only Sigmoid and Swish are implemented")
parser.add_argument("--sub_list", nargs='*', type=int,
                   help="Specify which subdomains to train.", default=[])
args = parser.parse_args()

# grid points for FD discretization
nx = args.nx
ny = args.ny

# number of subdomains in x and y directions
n_sub_x = args.n_sub_x
n_sub_y = args.n_sub_y

# size of latent dimension
intr_ld = args.intr_ld
intf_ld = args.intf_ld

# mask parameters
intr_row_nnz = args.intr_row_nnz
intf_row_nnz = args.intf_row_nnz
intr_row_shift = args.intr_row_shift
intf_row_shift = args.intf_row_shift

# number of snapshots used for training
nsnaps = args.nsnaps

print('Training DD-NM-ROM with options:')
print(f'(nx, ny) = ({nx}, {ny})')
print(f'(n_sub_x, n_sub_y) = ({n_sub_x}, {n_sub_y})')
print(f'interior state ROM dimension = {intr_ld}')
print(f'interface state ROM dimension = {intf_ld}')
print(f'interior row_nnz = {intr_row_nnz}')
print(f'interface row_nnz = {intf_row_nnz}')
print(f'interior row_shift = {intr_row_shift}')
print(f'interface row_shift = {intf_row_shift}\n')
sys.stdout.flush()

# Set print option
np.set_printoptions(threshold=sys.maxsize)

# set plotting options
plt.rc('font', size=20)
plt.rcParams['text.usetex'] = True

# Choose device that is not being used
# gpu_ids = "0"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
sys.stdout.flush()

# define constant parameters for PDE
x_lim   = [-1.0, 1.0]
y_lim   = [0.0, 0.05]

a1, lam = 7692.5384, 21.9230
viscosity = 1e-1
n_sub   = n_sub_x*n_sub_y

print('Loading snapshot data...')
sys.stdout.flush()

# load residual data
file = f'./data/residual_nx_{nx}_ny_{ny}_mu_{viscosity}_Nsamples_400.p'
data = pickle.load(open(file, 'rb'))
Mu_res = data['parameters']
residual_data = data['residuals']

# load snapshot data
Ntotal = 6400 if nx in [240, 480] else 4200
file = f'./data/snapshot_nx_{nx}_ny_{ny}_mu_{viscosity}_Nsamples_{Ntotal}.p'
data = pickle.load(open(file, 'rb'))
Mu = data['parameters']
snapshot_data = data['snapshots']

if nsnaps < len(snapshot_data):
    idx = np.linspace(0, len(snapshot_data), num=nsnaps, endpoint=False, dtype=int)
    snapshot_data = list(np.vstack(snapshot_data)[idx])
else:
    nsnaps = len(snapshot_data)

print('Data loaded!')
print(f'Training ROM using {nsnaps} snapshots.\n')
sys.stdout.flush()

net_folder0 = f'./trained_nets/nx_{nx}_ny_{ny}_mu_{viscosity}_{n_sub_x}x_by_{n_sub_y}y/'
fig_folder0 = f'./figures/loss_hist/nx_{nx}_ny_{ny}_mu_{viscosity}_{n_sub_x}x_by_{n_sub_y}y/'
for folder in [net_folder0, fig_folder0]:
    if not os.path.exists(folder):
        os.mkdir(folder)

# compute FOM for given a1 and lambda
def u_exact(x, y):
    phi = a1 + a1*x + (np.exp(lam*(x-1.0)) + np.exp(-lam*(x - 1.0)))*np.cos(lam*y)
    val = -2.0*viscosity*(a1 + lam*(np.exp(lam*(x-1.0)) \
          - np.exp(-lam*(x - 1.0)))*np.cos(lam*y))/phi
    return val
def v_exact(x, y):
    phi = a1 + a1*x + (np.exp(lam*(x-1.0)) + np.exp(-lam*(x - 1.0)))*np.cos(lam*y)
    val =  2.0*viscosity*(lam*(np.exp(lam*(x-1.0)) \
           + np.exp(-lam*(x - 1.0)))*np.sin(lam*y))/phi
    return val

print('Computing DD FOM...')
sys.stdout.flush()
# generate Burgers FOM on full domain
fom = Burgers2D(nx, ny, x_lim, y_lim, viscosity, u_exact, v_exact)

# compute Burgers DD FOM
ddmdl = DD_model(fom, n_sub_x, n_sub_y)
print('DD FOM computed!')
sys.stdout.flush()

# solve DD model 
ndof_fom = 2*np.sum([s.n_interior+s.n_interface for s in ddmdl.subdomain]) \
            + ddmdl.subdomain[0].constraint_mat.shape[0]
w0 = np.zeros(ndof_fom)

print('Solving DD model...')
sys.stdout.flush()
u_dd_fom, v_dd_fom, u_intr, v_intr, u_intf, v_intf, lam_fom, fom_time, itr = ddmdl.solve(w0, tol=1e-3, 
                                                                           maxit=100,
                                                                           print_hist=False)
print('DD model solved!\n')
sys.stdout.flush()

# autoencoder architecture parameters
latent_dim_dict     = {'interior' : intr_ld*np.ones(n_sub, dtype=int),
                       'interface': intf_ld*np.ones(n_sub, dtype=int)}
row_nnz_dict        = {'interior' : intr_row_nnz*np.ones(n_sub, dtype=int), 
                       'interface': intf_row_nnz*np.ones(n_sub, dtype=int)}
row_shift_dict      = {'interior' : intr_row_shift*np.ones(n_sub, dtype=int), 
                       'interface': intr_row_shift*np.ones(n_sub, dtype=int)}

if args.act_type in ['swish', 'Swish']:
    act_type = 'Swish'
else:
    act_type = 'Sigmoid'
print(f'Training autoencoders with {act_type} activation.')

loss_type = 'AbsMSE'       # RelMSE or AbsMSE

# select which subdomains and which states to train AEs for 
if args.intr_only and not args.intf_only:
    print('Only training for interior states')   
    phases = ['interior']
elif args.intf_only and not args.intr_only:
    print('Only training interface states')
    phases = ['interface']
else:
    print('Training interior and interface NNs.')
    phases = ['interior', 'interface']

if len(args.sub_list)>0:
    sub_list = args.sub_list  
else: 
    sub_list = np.arange(n_sub, dtype=int)
    
print(f'Training subdomains: {sub_list}')
sys.stdout.flush()     

# training parameters
epochs = 2000
n_epochs_print = 200
early_stop_patience = 300
batch_size = args.batch
lr_patience = 50
            
# separate snapshots to subdomains
interior, interface, residual = separate_snapshots(ddmdl, snapshot_data, residuals=residual_data)

# training for each subdomain and for interface and interiors
for phase in phases:   
    for sub in sub_list:
        latent_dim = latent_dim_dict[phase][sub]
        row_nnz    = row_nnz_dict[phase][sub]
        row_shift  = row_shift_dict[phase][sub]
    
        if phase == 'interior': # train interior roms
            snapshots = torch.from_numpy(np.vstack(interior[sub])).type(torch.float32)
            state     = torch.from_numpy(np.hstack([u_intr[sub], v_intr[sub]])).type(torch.float32).to(device)
            
        elif phase == 'interface': # train interface roms
            snapshots = torch.from_numpy(np.vstack(interface[sub])).type(torch.float32)
            state     = torch.from_numpy(np.hstack([u_intf[sub], v_intf[sub]])).type(torch.float32).to(device)
        
        # instantiate autoencoder
        autoencoder = Autoencoder(snapshots, latent_dim, row_nnz, row_shift, device, 
                                  act_type=act_type, 
                                  test_prop=0.1, 
                                  seed=None, 
                                  lr=1e-3, 
                                  lr_patience=lr_patience, 
                                  loss = loss_type)

        # train autoencoder
        net_folder = net_folder0 + f'sub_{sub+1}of{n_sub}/'
        if not os.path.exists(net_folder):
            os.mkdir(net_folder)
        net_folder += f'{phase}/'
        if not os.path.exists(net_folder):
            os.mkdir(net_folder)
            
        filename = net_folder + \
                    f'ld_{latent_dim}_rnnz_{row_nnz}_rshift_{row_shift}_'  + \
                    f'{act_type}_batch_{batch_size}_{loss_type}loss_{nsnaps}snaps.p'

        print(f'Training {phase} subdomain {sub}:')
        print(f'NN parameters: latent_dim={latent_dim}, row_nnz={row_nnz}, row_shift={row_shift}, batch_size={batch_size}, {act_type}')
        
        train_hist_dict, autoencoder_dict = autoencoder.train(batch_size, epochs,  
                                                             n_epochs_print=n_epochs_print, 
                                                             early_stop_patience=early_stop_patience, 
                                                             save_net=True, 
                                                             filename=filename)
        
        # compute autoencoder error on prediction 
        pred      = autoencoder.forward(state)
        rel_error = torch.norm(state-pred)/torch.norm(state)
        print(f'\na={a1}, lambda={lam} predictive case:')
        print(f'Prediction relative error = {rel_error:1.4e}\n')
        
        fig_folder = fig_folder0 + f'sub_{sub+1}of{n_sub}/'
        if not os.path.exists(fig_folder):
            os.mkdir(fig_folder)
        fig_folder += f'{phase}/'
        if not os.path.exists(fig_folder):
            os.mkdir(fig_folder)
            
        figname = fig_folder+\
                  f'ld_{latent_dim}_rnnz_{row_nnz}_rshift_{row_shift}_'+\
                  f'{act_type}_batch_{batch_size}_{loss_type}loss_{nsnaps}snaps.png'
        
        plt.figure(figsize=(10,8))
        plt.semilogy(train_hist_dict['train_loss_hist'], label='Train')
        plt.semilogy(train_hist_dict['test_loss_hist'], '--', label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig(figname)
