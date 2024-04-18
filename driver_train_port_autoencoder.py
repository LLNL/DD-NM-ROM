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
parser.add_argument("--port_ld", default=2, type=int, 
                   help="ROM dimension for port states")
parser.add_argument("--port_row_nnz", default=3, type=int, 
                   help="Number of nonzeros per row (col) in port decoder (encoder) mask")
parser.add_argument("--port_row_shift", default=3, type=int, 
                   help="Row (col) shift in port decoder (encoder) mask")
parser.add_argument("--nsnaps", default=6400, type=int, 
                   help="Number of snapshots used for training")
parser.add_argument("--batch", default=32, type=int, 
                   help="batch size for training")
parser.add_argument("--act_type", default = 'Sigmoid',
                   help="Activation type. Only Sigmoid and Swish are implemented")
parser.add_argument("--port_list", nargs='*', type=int,
                   help="Specify which ports to train.", default=[])
args = parser.parse_args()

# grid points for FD discretization
nx = args.nx
ny = args.ny

# number of subdomains in x and y directions
n_sub_x = args.n_sub_x
n_sub_y = args.n_sub_y

# size of latent dimension
port_ld = args.port_ld

# mask parameters
port_row_nnz = args.port_row_nnz
port_row_shift = args.port_row_shift

# number of snapshots used for training
nsnaps = args.nsnaps

print('Training DD-NM-ROM with options:')
print(f'(nx, ny) = ({nx}, {ny})')
print(f'(n_sub_x, n_sub_y) = ({n_sub_x}, {n_sub_y})')
print(f'port state ROM dimension = {port_ld}')
print(f'port row_nnz = {port_row_nnz}')
print(f'port row_shift = {port_row_shift}')
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

if args.act_type in ['swish', 'Swish']:
    act_type = 'Swish'
else:
    act_type = 'Sigmoid'
print(f'Training autoencoders with {act_type} activation.')
sys.stdout.flush()

loss_type = 'AbsMSE'       # RelMSE or AbsMSE

# training parameters
epochs = 2000
n_epochs_print = 200
early_stop_patience = 300
batch_size = args.batch
lr_patience = 50
n_ports = len(ddmdl.ports)

if len(args.port_list)>0:
    port_list = args.port_list
    print(f'Training ports:')
    for i in port_list: print(f'P({i}) = {ddmdl.ports[i]}')
    sys.stdout.flush()       
else: 
    port_list = np.arange(n_ports, dtype=int)

# training for each subdomain and for interface and interiors
for i in port_list:
    port       = ddmdl.ports[i]
    port_ind   = ddmdl.port_dict[port]
    port_snaps = np.vstack([np.concatenate([snap[port_ind], snap[port_ind+ddmdl.nxy]]) for snap in snapshot_data])
    port_snaps = torch.from_numpy(port_snaps).type(torch.float32)
    state      = torch.from_numpy(np.hstack([u_dd_fom[port_ind], v_dd_fom[port_ind]])).type(torch.float32).to(device)

    # instantiate autoencoder
    latent_dim  = max(min(port_ld, port_snaps.shape[1]-1), 1)
#     latent_dim  = port_ld
    autoencoder = Autoencoder(port_snaps, latent_dim, port_row_nnz, port_row_shift, device, 
                              act_type=act_type, 
                              test_prop=0.1, 
                              seed=None, 
                              lr=1e-3, 
                              lr_patience=lr_patience, 
                              loss = loss_type)

    # train autoencoder
    net_folder = net_folder0 + f'port_{i+1}of{n_ports}/'
    if not os.path.exists(net_folder):
        os.mkdir(net_folder)

    filename = net_folder + \
                f'ld_{latent_dim}_rnnz_{port_row_nnz}_rshift_{port_row_shift}_'  + \
                f'{act_type}_batch_{batch_size}_{loss_type}loss_{nsnaps}snaps.p'

    print(f'Training port P({i})={port}:')
    print(f'NN parameters: latent_dim={latent_dim}, row_nnz={port_row_nnz}, row_shift={port_row_shift}, batch_size={batch_size}, {act_type}')

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

    fig_folder = fig_folder0 + f'port_{i+1}of{n_ports}/'
    if not os.path.exists(fig_folder):
        os.mkdir(fig_folder)

    figname = fig_folder+\
              f'ld_{latent_dim}_rnnz_{port_row_nnz}_rshift_{port_row_shift}_'+\
              f'{act_type}_batch_{batch_size}_{loss_type}loss_{nsnaps}snaps.png'

    plt.figure(figsize=(10,8))
    plt.semilogy(train_hist_dict['train_loss_hist'], label='Train')
    plt.semilogy(train_hist_dict['test_loss_hist'], '--', label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(figname)
