# generate_data
# Generate snapshot data for the 2D Burgers equation for a number of parameters.  
# Author: Alejandro Diaz

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.sparse as sp
from utils.Burgers2D_probgen import Burgers2D
import dill as pickle
import sys, os
import argparse

save_dir = './data/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
parser = argparse.ArgumentParser()
parser.add_argument("--nx", default=240, type=int, 
                   help="number of grid points in x-direction")
parser.add_argument("--ny", default=12, type=int, 
                   help="number of grid points in y-direction")
parser.add_argument("--na", default=20, type=int, 
                   help="number of grid points in a-direction on parameter grid")
parser.add_argument("--nlam", default=80, type=int, 
                   help="number of grid points in lambda-direction on parameter grid")
parser.add_argument("--viscosity", default=1e-1, type=float, 
                    help="viscosity for 2D Burgers' problem")
parser.add_argument("--maxit", default=80, type=int, 
                    help="maximum number of iterations for Newton solver")
parser.add_argument("--tol", default=1e-8, type=float, 
                    help="absolute tolerance for Newton solver")
args = parser.parse_args()

# PDE discretization parameters
x_lim = [-1.0, 1.0]
y_lim = [0.0, 0.05]
nx, ny    = args.nx, args.ny
viscosity = args.viscosity

# parameter space
a1_lim     = [1.0, 10000.0]
lam_lim    = [5.0, 25.0]
na, nlam   = args.na, args.nlam

print('\nComputing state snapshots with the following parameters:')
print(f'nx        = {nx}')
print(f'ny        = {ny}')
print(f'viscosity = {viscosity}')
print(f'na        = {na}')
print(f'nlam      = {nlam}')

print('\nParameters for Newton solver:')
print(f'maxit = {args.maxit}')
print(f'tol   = {args.tol}')
sys.stdout.flush()

# Generates parameters
a1_vals  = np.linspace(a1_lim[0], a1_lim[1], na)
lam_vals = np.linspace(lam_lim[0], lam_lim[1], nlam)
A1, Lam  = np.meshgrid(a1_vals, lam_vals)
Mu = np.vstack([A1.flatten(), Lam.flatten()]).T

# compute solution for each pair of parameters
u0 = np.zeros(nx*ny)
v0 = np.zeros(nx*ny)
snapshots = []
for mu in Mu:
    # sets current parameters
    a1, lam = mu
    
    print(f'\nComputing solutions for a1 = {a1}, lam={lam}...')
    # exact u and v solutions
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
    
    # generate Burgers FOM
    mdl = Burgers2D(nx, ny, x_lim, y_lim, viscosity, u_exact, v_exact)
    
    # solve for u and v
    u, v, res_vecs = mdl.solve(u0, v0, tol=args.tol, maxit=args.maxit, print_hist=False)
    
    # store snapshots and intermediate Newton residuals
    snapshots.append(np.concatenate((u, v)))
    sys.stdout.flush()

# save data to file
save_dict = {'parameters': Mu,
             'snapshots': snapshots}
file = save_dir+f'snapshot_nx_{nx}_ny_{ny}_mu_{viscosity}_Nsamples_{len(Mu)}.p'
pickle.dump(save_dict, open(file,'wb'))