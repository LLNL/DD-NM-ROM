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
from utils.NM_ROM import separate_snapshots
from utils.autoencoder import Autoencoder, Encoder, Decoder

# Parser for command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--nx", default=480, type=int, 
                   help="number of grid points in x-direction")
parser.add_argument("--ny", default=24, type=int, 
                   help="number of grid points in y-direction")
parser.add_argument("--ld", default=8, type=int, 
                   help="ROM dimension")
parser.add_argument("--row_nnz", default=5, type=int, 
                   help="Number of nonzeros per row (col) in decoder (encoder) mask")
parser.add_argument("--row_shift", default=5, type=int, 
                   help="Row (col) shift in decoder (encoder) mask")
parser.add_argument("--nsnaps", default=6400, type=int, 
                   help="Number of snapshots used for training")
parser.add_argument("--batch", default=32, type=int, 
                   help="batch size for training")
parser.add_argument("--act_type", default = 'Sigmoid',
                   help="Activation type. Only Sigmoid and Swish are implemented")
parser.add_argument("--dense_encoder", action='store_true',
                   help="Pass to use dense encoder")
parser.add_argument("--enc_hidden", default=-1, type=float, 
                   help="Multiplier for determining dimension of hidden layer for dense encoder.")
args = parser.parse_args()

# grid points for FD discretization
nx = args.nx
ny = args.ny

# size of latent dimension
latent_dim = args.ld

# mask parameters
row_nnz   = args.row_nnz
row_shift = args.row_shift
dense_encoder = args.dense_encoder

# number of snapshots used for training
nsnaps = args.nsnaps

print('Training DD-NM-ROM with options:')
print(f'(nx, ny)            = ({nx}, {ny})')
print(f'(n_sub_x, n_sub_y)  = ({1}, {1})')
print(f'ROM state dimension = {latent_dim}')
print(f'row_nnz             = {row_nnz}')
print(f'row_shift           = {row_shift}')
print(f'dense encoder       = {dense_encoder}\n')
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

print('Loading snapshot data...')
sys.stdout.flush()

# load residual data
# file = f'./data/residual_nx_{nx}_ny_{ny}_mu_{viscosity}_Nsamples_400.p'
# data = pickle.load(open(file, 'rb'))
# Mu_res = data['parameters']
# residual_data = data['residuals']

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

net_folder0 = f'./trained_nets/nx_{nx}_ny_{ny}_mu_{viscosity}_{1}x_by_{1}y/'
fig_folder0 = f'./figures/loss_hist/nx_{nx}_ny_{ny}_mu_{viscosity}_{1}x_by_{1}y/'
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

print('Computing FOM...')
sys.stdout.flush()
# generate Burgers FOM on full domain
fom = Burgers2D(nx, ny, x_lim, y_lim, viscosity, u_exact, v_exact)

# generate Burgers FOM on full domain
print('Solving full domain model:')
sys.stdout.flush()
u_fom, v_fom, res_hist = fom.solve(np.zeros(fom.nxy), np.zeros(fom.nxy), tol=1e-8, print_hist=True)
state = torch.from_numpy(np.hstack([u_fom, v_fom])).type(torch.float32).to(device)

print('FOM solved!\n')
sys.stdout.flush()

# autoencoder architecture parameters
if args.act_type in ['swish', 'Swish']:
    act_type = 'Swish'
else:
    act_type = 'Sigmoid'
print(f'Training autoencoders with {act_type} activation.')

loss_type = 'AbsMSE'       # RelMSE or AbsMSE

# training parameters
epochs = 2000
n_epochs_print = 200
early_stop_patience = 300
batch_size = args.batch
lr_patience = 50

# training
snapshots = torch.from_numpy(np.vstack(snapshot_data)).type(torch.float32)

enc_hidden_mult = args.enc_hidden
encoder_hidden=-1 if enc_hidden_mult < 0 else np.round(enc_hidden_mult*snapshots.shape[1]).astype(int)

if dense_encoder:
    print(f'Encoder hidden = {encoder_hidden} (-1 uses 2*(dim of input))')

# instantiate autoencoder
autoencoder = Autoencoder(snapshots, latent_dim, row_nnz, row_shift, device, 
                          dense_encoder=dense_encoder,
                          encoder_hidden=encoder_hidden,
                          act_type=act_type, 
                          test_prop=0.1, 
                          seed=None, 
                          lr=1e-3, 
                          lr_patience=lr_patience, 
                          loss = loss_type)

# train autoencoder
net_folder = net_folder0 + f'sub_{1}of{1}/'
if not os.path.exists(net_folder):
    os.mkdir(net_folder)

de_str = '_dense_enc' if dense_encoder else ''
filename = net_folder + \
            f'ld_{latent_dim}_rnnz_{row_nnz}_rshift_{row_shift}_'  + \
            f'{act_type}_batch_{batch_size}_{loss_type}loss_{nsnaps}snaps{de_str}.p'

print(f'Training full domain autoencoder:')
print(f'NN parameters: latent_dim={latent_dim}, row_nnz={row_nnz}, row_shift={row_shift}, batch_size={batch_size}, {act_type}')

autoencoder_dict = autoencoder.train(batch_size, epochs,  
                                     n_epochs_print=n_epochs_print, 
                                     early_stop_patience=early_stop_patience, 
                                     save_net=True, 
                                     filename=filename)

# compute autoencoder error on prediction 
pred      = autoencoder.forward(state)
rel_error = torch.norm(state-pred)/torch.norm(state)
print(f'\na={a1}, lambda={lam} predictive case:')
print(f'Prediction relative error = {rel_error:1.4e}\n')

fig_folder = fig_folder0 + f'sub_{1}of{1}/'
if not os.path.exists(fig_folder):
    os.mkdir(fig_folder)

figname = fig_folder+\
          f'ld_{latent_dim}_rnnz_{row_nnz}_rshift_{row_shift}_'+\
          f'{act_type}_batch_{batch_size}_{loss_type}loss_{nsnaps}snaps{de_str}.png'

plt.figure(figsize=(10,8))
plt.semilogy(autoencoder_dict['train_loss_hist'], label='Train')
plt.semilogy(autoencoder_dict['test_loss_hist'], '--', label='Test')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.savefig(figname)
