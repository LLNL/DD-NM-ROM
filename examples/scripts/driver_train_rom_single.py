'''
Train autoencoders for DD-NM-ROM.
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
net_folder0 = args.save_dir + f'/trained_nets/{name}/single/'
fig_folder0 = args.save_dir + f'/figures/loss_hist/{name}/single/'
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
if args.intr_only:
  epochs = 6000
  n_epochs_print = 200
  early_stop_patience = epochs/2
  batch_size = args.batch
  lr_patience = 50*4
elif args.intf_only:
  epochs = 4000
  n_epochs_print = 200
  early_stop_patience = epochs/2
  batch_size = args.batch
  lr_patience = 50*2

# separate snapshots to subdomains
print("\nSplitting snapshots")
data = ddmdl.map_sol_on_elements(np.vstack(snapshot_data))
print("Snapshots splited!")

# training for each subdomain and for interface and interiors
for phase in phases:

  latent_dim = latent_dim_dict[phase][0]
  row_nnz    = row_nnz_dict[phase][0]
  row_shift  = row_shift_dict[phase][0]

  snapshots = torch.from_numpy(np.vstack(data[phase])).type(torch.float32)

  # train autoencoder
  net_folder = net_folder0 + f'/{phase}/'
  if not os.path.exists(net_folder):
    os.mkdir(net_folder)

  filename = net_folder + \
        f'ld_{latent_dim}_rnnz_{row_nnz}_rshift_{row_shift}_'  + \
        f'{act_type}_batch_{batch_size}_{loss_type}loss_{nsnaps}snaps.p'

  # instantiate autoencoder
  autoencoder = Autoencoder(snapshots, latent_dim, row_nnz, row_shift, device,
                act_type=act_type,
                test_prop=0.1,
                seed=None,
                lr=1e-3,
                lr_patience=lr_patience,
                loss=loss_type,
                filename=filename)

  print(f'Training {phase}:')
  print(f'NN parameters: latent_dim={latent_dim}, row_nnz={row_nnz}, row_shift={row_shift}, batch_size={batch_size}, {act_type}')

  autoencoder_dict = autoencoder.train(batch_size, epochs,
                              n_epochs_print=n_epochs_print,
                              early_stop_patience=early_stop_patience,
                              save_net=True)

  fig_folder = fig_folder0 + f'/{phase}/'
  if not os.path.exists(fig_folder):
    os.mkdir(fig_folder)

  figname = fig_folder+\
        f'ld_{latent_dim}_rnnz_{row_nnz}_rshift_{row_shift}_'+\
        f'{act_type}_batch_{batch_size}_{loss_type}loss_{nsnaps}snaps.png'

  plt.figure(figsize=(10,8))
  plt.semilogy(autoencoder_dict['train_loss_hist'], label='Train')
  plt.semilogy(autoencoder_dict['valid_loss_hist'], '--', label='Test')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.legend()
  plt.savefig(figname)
