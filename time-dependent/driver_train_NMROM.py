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
from utils.domain_decomposition import DD_FOM
from utils.autoencoder import Autoencoder, Encoder, Decoder
from utils.lsrom import assemble_snapshot_matrix

# Parser for command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--nsub_x", default=2, type=int, 
                   help="number of subdomains in x-direction")
parser.add_argument("--nsub_y", default=1, type=int, 
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

# parser.add_argument("--nsnaps", default=-1, type=int, 
#                    help="Number of snapshots used for training. -1 uses nt*len(mu_list)")

parser.add_argument("--epochs", default=2000, type=int, 
                    help="number of training epochs")
parser.add_argument("--batch", default=32, type=int, 
                   help="batch size for training")
parser.add_argument("--es_patience", default=300, type=int, 
                   help="early stopping patience")
parser.add_argument("--lr", default=1e-3, type=float, 
                   help="initial learning rate")
parser.add_argument("--lr_patience", default=50, type=int, 
                   help="learning rate adjustment patience")

parser.add_argument("--wd_intr", default=0, type=float, 
                   help="Weight decay for interior AEs, i.e. L2 regularization parameter.")
parser.add_argument("--wd_intf", default=0, type=float, 
                   help="Weight decay for interface AEs, i.e. L2 regularization parameter.")
parser.add_argument("--wd_port", default=0, type=float, 
                   help="Weight decay for interface AEs, i.e. L2 regularization parameter.")

parser.add_argument("--intr_act", default = 'Swish', type=str,
                   help="Interior state activation type. Only Sigmoid and Swish are implemented")
parser.add_argument("--intf_act", default = 'Swish', type=str,
                   help="Interface state activation type. Only Sigmoid and Swish are implemented")
parser.add_argument("--port_act", default = 'Swish', type=str,
                   help="Interface state activation type. Only Sigmoid and Swish are implemented")
parser.add_argument("--loss", default='AbsMSE', type=str,
                    help="loss type. 'AbsMSE' or 'RelMSE.'")

parser.add_argument("--sub_list", nargs='*', type=int,
                   help="Specify which subdomains to train.", default=[])
parser.add_argument("--port_list", nargs='*', type=int,
                   help="Specify which ports to train.", default=[])
parser.add_argument("--comps", nargs='*', type=str,
                   help="Specify which subdomains to train.", default=['interior','interface','port'])
args = parser.parse_args()

# parameters for physical domain and FD discretization
x_lim = [0, 1]
y_lim = [0, 1]
t_lim = [0, 2]
nx, ny, nt = args.nx, args.ny, args.nt
viscosity = args.viscosity
nsub_x, nsub_y = args.nsub_x, args.nsub_y
n_sub   = nsub_x*nsub_y
mu_list = args.mu_list

# size of latent dimension
intr_size, intf_size, port_size = args.intr_size, args.intf_size, args.port_size

# mask parameters
intr_row_nnz = args.intr_row_nnz
intf_row_nnz = args.intf_row_nnz
port_row_nnz = args.port_row_nnz
intr_row_shift = args.intr_row_shift
intf_row_shift = args.intf_row_shift
port_row_shift = args.port_row_shift
wd_intr     = args.wd_intr
wd_intf     = args.wd_intf
wd_port     = args.wd_port

# autoencoder/training parameters
epochs         = args.epochs
batch_size     = args.batch
n_epochs_print = 200
es_patience    = args.es_patience
lr             = args.lr
lr_patience    = args.lr_patience
comps          = args.comps
loss_type      = args.loss

# # number of snapshots used for training
# nsnaps = nt*len(mu_list) if args.nsnaps < 0 else args.nsnaps

print('Training DD NM-ROM with options:')
print(f'(nx, ny, nt)         = ({nx}, {ny}, {nt})')
print(f'viscosity            = {viscosity:1.2e}')
print(f'(nsub_x, nsub_y)     = ({nsub_x}, {nsub_y})\n')

print(f'interior ROM size    = {intr_size}')
print(f'interior row_nnz     = {intr_row_nnz}')
print(f'interior row_shift   = {intr_row_shift}')
print(f'interior activation  = {args.intr_act}')
print(f'intr wd              = {wd_intr}\n')

print(f'interface ROM size   = {intf_size}')
print(f'interface row_nnz    = {intf_row_nnz}')
print(f'interface row_shift  = {intf_row_shift}')
print(f'interface activation = {args.intf_act}')
print(f'intf wd              = {wd_intf}\n')

print(f'port ROM size        = {port_size}')
print(f'port row_nnz         = {port_row_nnz}')
print(f'port row_shift       = {port_row_shift}')
print(f'port activation      = {args.port_act}')
print(f'port wd              = {wd_port}\n')

sys.stdout.flush()

print('Autoencoder training parameters:')
print(f'epochs              = {epochs}')
print(f'batch size          = {batch_size}')
print(f'early stop patience = {es_patience}')
print(f'Initial LR          = {lr}')
print(f'LR patience         = {lr_patience}')
print(f'Loss type           = {loss_type}\n')

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
fig_dir0  = 'figures/'
fig_dir1  = fig_dir0 + f'nx_{nx}_ny_{ny}_nt_{nt}_visc_{viscosity}/'
fig_dir2  = fig_dir1 + 'loss_hist/'
fig_dir   = fig_dir2 + f'DD_{nsub_x}x_by_{nsub_y}y/'
data_dir0 = 'data/'
data_dir1 = data_dir0 + f'nx_{nx}_ny_{ny}_nt_{nt}_visc_{viscosity}/'
# data_dir  = data_dir1 + f'DD_{nsub_x}x_by_{nsub_y}y/'
net_dir0  = 'trained_nets/'
net_dir1  = net_dir0 + f'nx_{nx}_ny_{ny}_nt_{nt}_visc_{viscosity}/'
net_dir   = net_dir1 + f'DD_{nsub_x}x_by_{nsub_y}y/'
for d in [fig_dir0, fig_dir1, fig_dir2, fig_dir, data_dir0, data_dir1, net_dir0, net_dir1, net_dir]:
    if not os.path.exists(d): os.mkdir(d)

# load snapshot data
print('\nLoading snapshot data...')
sys.stdout.flush()
data = assemble_snapshot_matrix(data_dir1, mu_list)
print('Done!\n')
sys.stdout.flush()

# Set device and default dtype to float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
print(f"Using device: {device}")
sys.stdout.flush()

# autoencoder architecture parameters
n_ports = len(ddfom.ports)
latent_dim_dict     = {'interior' : intr_size*np.ones(n_sub, dtype=int),
                       'interface': intf_size*np.ones(n_sub, dtype=int), 
                       'port'     : port_size*np.ones(n_ports, dtype=int)}
row_nnz_dict        = {'interior' : intr_row_nnz*np.ones(n_sub, dtype=int), 
                       'interface': intf_row_nnz*np.ones(n_sub, dtype=int), 
                       'port'     : port_row_nnz*np.ones(n_ports, dtype=int)}
row_shift_dict      = {'interior' : intr_row_shift*np.ones(n_sub, dtype=int), 
                       'interface': intr_row_shift*np.ones(n_sub, dtype=int), 
                       'port'     : port_row_shift*np.ones(n_ports, dtype=int)}

activations = {'interior' : args.intr_act,
               'interface': args.intf_act,
               'port'     : args.port_act}
wd_dict     = {'interior' : wd_intr, 
               'interface': wd_intf, 
               'port'     : wd_port}

# select which subdomains and which states to train AEs for 
if len(args.sub_list)>0:
    sub_list = args.sub_list  
else: 
    sub_list = np.arange(n_sub, dtype=int)
if len(args.port_list)>0:
    port_list = args.port_list  
else: 
    port_list = np.arange(len(ddfom.ports), dtype=int)
    
print(f'\nTraining components: {comps}')
print(f'Training subdomains:   {sub_list}')
print(f'Training ports:        {port_list}\n')
sys.stdout.flush()     
            
# training for each subdomain and for interface and interiors
for comp in comps:
    act_type   = activations[comp]
    if comp == 'port':
        for j in port_list: #, p in enumerate(ddfom.ports):
            p = ddfom.ports[j]
            latent_dim = latent_dim_dict[comp][j]
            row_nnz    = row_nnz_dict[comp][j]
            row_shift  = row_shift_dict[comp][j]

            print(f'Port {j} = {p}...')
            sys.stdout.flush()

            snapshots = torch.from_numpy(np.vstack([data[ddfom.port_dict[p]], 
                                                    data[ddfom.port_dict[p]+ddfom.nxy]]).T)#.float()
            latent_dim = np.min([latent_dim, snapshots.shape[1]])
            
            save_dir = net_dir + f'port_{j+1}of{n_ports}/'
            if not os.path.exists(save_dir): os.mkdir(save_dir)

            filename = save_dir + f'ld_{latent_dim}_rnnz_{row_nnz}_rshift_{row_shift}_'\
                                + f'{act_type}_batch_{batch_size}_{loss_type}loss_wd{wd_dict[comp]}.p'

            autoencoder = Autoencoder(snapshots, latent_dim, row_nnz, row_shift, device, 
                                      act_type=act_type, 
                                      test_prop=0.1, 
                                      seed=None, 
                                      lr=lr, 
                                      lr_patience=lr_patience, 
                                      weight_decay=wd_dict[comp],
                                      loss = loss_type)

            print(f'NN parameters:')
            print(f'latent_dim = {latent_dim}')
            print(f'row_nnz    = {row_nnz}')
            print(f'row_shift  = {row_shift}')
            print(f'batch_size = {batch_size}')
            print(f'activation = {act_type}')

            train_hist_dict, autoencoder_dict = autoencoder.train(batch_size, 
                                                                  epochs,  
                                                                  n_epochs_print=n_epochs_print, 
                                                                  early_stop_patience=es_patience, 
                                                                  save_net=True, 
                                                                  filename=filename)

            fig_folder = fig_dir + f'port_{j+1}of{n_ports}/'
            if not os.path.exists(fig_folder): os.mkdir(fig_folder)

            figname = fig_folder+\
                      f'ld_{latent_dim}_rnnz_{row_nnz}_rshift_{row_shift}_'+\
                      f'{act_type}_batch_{batch_size}_{loss_type}loss_wd{wd_dict[comp]}.png'

            plt.figure(figsize=(10,8))
            plt.semilogy(train_hist_dict['train_loss_hist'], label='Train')
            plt.semilogy(train_hist_dict['test_loss_hist'], '--', label='Test')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.savefig(figname)
    
    else:
        for sub in sub_list:
            latent_dim = latent_dim_dict[comp][sub]
            row_nnz    = row_nnz_dict[comp][sub]
            row_shift  = row_shift_dict[comp][sub]
            s = ddfom.subdomain[sub]
            
            if comp == 'interior': # train interior roms
                snapshots = torch.from_numpy(np.vstack([data[s.interior.indices],
                                                        data[s.interior.indices+ddfom.nxy]]).T)#.float()

            elif comp == 'interface': # train interface roms
                snapshots = torch.from_numpy(np.vstack([data[s.interface.indices],
                                                        data[s.interface.indices+ddfom.nxy]]).T)#.float()

            # instantiate autoencoder
            autoencoder = Autoencoder(snapshots, latent_dim, row_nnz, row_shift, device, 
                                      act_type=act_type, 
                                      test_prop=0.1, 
                                      seed=None, 
                                      lr=lr, 
                                      lr_patience=lr_patience, 
                                      weight_decay=wd_dict[comp],
                                      loss = loss_type)

            # train autoencoder
            save_dir = net_dir + f'sub_{sub+1}of{n_sub}/'
            if not os.path.exists(save_dir): os.mkdir(save_dir)
            save_dir += f'{comp}/'
            if not os.path.exists(save_dir): os.mkdir(save_dir)

            filename = save_dir + \
                        f'ld_{latent_dim}_rnnz_{row_nnz}_rshift_{row_shift}_'  + \
                        f'{act_type}_batch_{batch_size}_{loss_type}loss_wd{wd_dict[comp]}.p'

            print(f'Training {comp} subdomain {sub}:')
            print(f'NN parameters:')
            print(f'latent_dim = {latent_dim}')
            print(f'row_nnz    = {row_nnz}')
            print(f'row_shift  = {row_shift}')
            print(f'batch_size = {batch_size}')
            print(f'activation = {act_type}')

            train_hist_dict, autoencoder_dict = autoencoder.train(batch_size, 
                                                                  epochs,  
                                                                  n_epochs_print=n_epochs_print, 
                                                                  early_stop_patience=es_patience, 
                                                                  save_net=True, 
                                                                  filename=filename)
            
            fig_folder = fig_dir + f'sub_{sub+1}of{n_sub}/'
            if not os.path.exists(fig_folder): os.mkdir(fig_folder)
            fig_folder += f'{comp}/'
            if not os.path.exists(fig_folder): os.mkdir(fig_folder)

            figname = fig_folder+\
                      f'ld_{latent_dim}_rnnz_{row_nnz}_rshift_{row_shift}_'+\
                      f'{act_type}_batch_{batch_size}_{loss_type}loss_wd{wd_dict[comp]}.png'

            plt.figure(figsize=(10,8))
            plt.semilogy(train_hist_dict['train_loss_hist'], label='Train')
            plt.semilogy(train_hist_dict['test_loss_hist'], '--', label='Test')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.savefig(figname)
