'''
Train port autoencoders for DD-NM-ROM.
'''

import sys
with open("./../../PATHS.txt") as file:
  paths = file.read().splitlines()
sys.path.extend(paths)

import os
import torch
import argparse
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt

from dd_nm_rom.rom.nonlinear import Autoencoder
from dd_nm_rom.fom import Burgers2D, DDBurgers2D, Burgers2DExact

# Parser for command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--res_file", type=str)
parser.add_argument("--snap_file", type=str)
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
parser.add_argument(
  "--save_dir", default='./data/', type=str,
  help="path to saving folder"
)
parser.add_argument(
  "--gpu_idx", default=0, type=int, choices=[0,1,2,3],
  help="GPU index"
)
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
plt.rcParams['text.usetex'] = False

# Choose device that is not being used
# gpu_ids = "0"
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids

# Set device
device = torch.device(f"cuda:{args.gpu_idx}" if torch.cuda.is_available() else "cpu")
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
data = pickle.load(open(args.res_file, 'rb'))
Mu_res = data['parameters']
residual_data = data['residuals']

# load snapshot data
data = pickle.load(open(args.snap_file, 'rb'))
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

name = f"nx_{nx}_ny_{ny}_mu_{viscosity}_{n_sub_x}x_by_{n_sub_y}y"
net_folder0 = args.save_dir + f'/trained_nets/{name}/single/ports/'
fig_folder0 = args.save_dir + f'/figures/loss_hist/{name}/single/ports/'
for folder in [net_folder0, fig_folder0]:
  os.makedirs(folder, exist_ok=True)

# compute FOM for given a1 and lambda
fom_ex = Burgers2DExact()
fom_ex.set_params(a1, lam, viscosity)

print('Computing DD FOM...')
sys.stdout.flush()
# generate Burgers FOM on full domain
fom = Burgers2D(nx, ny, x_lim, y_lim, viscosity)
fom.set_bc(fom_ex.u, fom_ex.v)

# compute Burgers DD FOM
ddmdl = DDBurgers2D(fom, n_sub_x, n_sub_y)
ddmdl.set_bc()

print('DD FOM computed!')
sys.stdout.flush()

print('Solving DD model...')
sys.stdout.flush()
uv_dd, lambdas = ddmdl.solve(x0=np.zeros(ddmdl.get_ndof()), tol=5e-4, maxit=100, verbose=False)

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
epochs = 4000
n_epochs_print = 200
early_stop_patience = epochs
batch_size = args.batch
lr_patience = 50*2


# use one autoencoder for all ports with the same dimension
autoencoder_to_ports = {}
for p in ddmdl.ports:
    port_dim = ddmdl.port_to_nodes[ddmdl.ports[p]].size
    if port_dim not in autoencoder_to_ports:
        autoencoder_to_ports[port_dim] = []
    autoencoder_to_ports[port_dim].append(p)

# training for each subdomain and for interface and interiors
for i, (port_dim, ports) in enumerate(autoencoder_to_ports.items()):
    port_snaps = []
    for p in ports:
        port_ind_p   = ddmdl.port_to_nodes[p]
        port_snaps_p = np.vstack([np.concatenate([snap[port_ind_p], snap[port_ind_p+ddmdl.nxy]]) for snap in snapshot_data])
        port_snaps.append(port_snaps_p)
    port_snaps = torch.from_numpy(np.vstack(port_snaps)).type(torch.float32)

    # instantiate autoencoder
    latent_dim  = max(min(port_ld, port_snaps.shape[1]-1), 1)

    # train autoencoder
    net_folder = net_folder0 + f'port_{i+1}_dim_{port_dim}/'
    os.makedirs(net_folder, exist_ok=True)

    filename = net_folder + \
                f'ld_{latent_dim}_rnnz_{port_row_nnz}_rshift_{port_row_shift}_'  + \
                f'{act_type}_batch_{batch_size}_{loss_type}loss_{nsnaps}snaps.p'

    # instantiate autoencoder
    autoencoder = Autoencoder(port_snaps, latent_dim, port_row_nnz, port_row_shift, device,
                              min_dim=5,
                              act_type=act_type,
                              test_prop=0.1,
                              seed=None,
                              lr=1e-3,
                              lr_patience=lr_patience,
                              loss=loss_type,
                              filename=filename)

    if not autoencoder.trainable:

        print(f'\n\nNot training ports with size of {port_dim}')

    else:

      print(f'\n\nTraining ports with size of {port_dim}:')
      print(f'NN parameters: latent_dim={latent_dim}, row_nnz={port_row_nnz}, row_shift={port_row_shift}, batch_size={batch_size}, {act_type}')

      autoencoder_dict = autoencoder.train(batch_size, epochs,
                                                          n_epochs_print=n_epochs_print,
                                                          early_stop_patience=early_stop_patience,
                                                          save_net=True)


      fig_folder = fig_folder0 + f'port_{i+1}_dim_{port_dim}/'
      os.makedirs(fig_folder, exist_ok=True)

      figname = fig_folder+\
                f'ld_{latent_dim}_rnnz_{port_row_nnz}_rshift_{port_row_shift}_'+\
                f'{act_type}_batch_{batch_size}_{loss_type}loss_{nsnaps}snaps.png'

      plt.figure(figsize=(10,8))
      plt.semilogy(autoencoder_dict['train_loss_hist'], label='Train')
      plt.semilogy(autoencoder_dict['valid_loss_hist'], '--', label='Test')
      plt.xlabel('Epoch')
      plt.ylabel('MSE Loss')
      plt.legend()
      plt.savefig(figname)
