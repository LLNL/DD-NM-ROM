import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from utils.Burgers2D_probgen import Burgers2D
import sys, os
import argparse
import dill as pickle

# make directories for figures and data
data_dir = './data/'
fig_dir0 = './figures/'
for d in [data_dir, fig_dir0]:
    if not os.path.exists(d): os.mkdir(d)

parser = argparse.ArgumentParser()
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
parser.add_argument("--tol", default=1e-9, type=float, 
                    help="absolute tolerance for Newton solver")
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

print('\nComputing state snapshots with the following parameters:')
print(f'nx        = {nx}')
print(f'ny        = {ny}')
print(f'nt        = {nt}')
print(f'viscosity = {viscosity}')
print(f'mu_list   = {mu_list}')

print('\nParameters for Newton solver:')
print(f'maxit = {args.maxit}')
print(f'tol   = {args.tol}')
sys.stdout.flush()

# initialize model
print('\nInitializing Burgers model...')
sys.stdout.flush()
mdl = Burgers2D(nx, ny, x_lim, y_lim, viscosity)
print('Done!')

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
for d in [fig_dir1, data_dir1]:
    if not os.path.exists(d): os.mkdir(d)

# frame updater for animation
X, Y = np.meshgrid(mdl.xx, mdl.yy)
def update_frame(Z, i, cb_label):
        plt.clf()
        plt.pcolormesh(X, Y, Z[i], cmap='viridis', shading='auto', vmin=Z.min(), vmax=Z.max()) 
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        cb = plt.colorbar(orientation='vertical', label=cb_label)
        return plt

for mu in mu_list:   
    print(f'\nRunning for mu = {mu}:')
    sys.stdout.flush()

    # make figure directories
    fig_dir2 = fig_dir1 + f'mu_{mu}/'
    fom_figs = fig_dir2 + 'fom/'
    for d in [fig_dir1, fig_dir2, fom_figs]:
        if not os.path.exists(d): os.mkdir(d)
    
    # set initial condition
    mdl.set_initial(lambda xy: u0(xy, mu), lambda xy: v0(xy, mu))
    
    # solve Burgers equation
    print('\nSolve Burgers equation with backward Euler:')
    sys.stdout.flush()
    uu, vv, res_hist, step_hist, runtime, flag = mdl.solve(t_lim, nt, tol=args.tol, maxit=args.maxit, print_hist=False)
    print('Done!') 
    sys.stdout.flush()
    
    # store snapshots and residuals
    print('Saving data...')
    sys.stdout.flush()
    uv = {'solution': np.hstack([uu, vv]), 'runtime': runtime}
    pickle.dump(uv, open(data_dir1+f'mu_{mu}_uv_state.p', 'wb'))
    print('Done!')
    print(f'Runtime = {runtime:1.5e} seconds')
    sys.stdout.flush()
    
    # save gifs of solutions
    print('\nGenerating gif for u state...')
    sys.stdout.flush()
    UU = uu.reshape(nt+1, mdl.ny, mdl.nx)
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, lambda i: update_frame(UU, i, '$u(x,y)$'), frames=nt+1, interval=1)
    filename = fom_figs + 'u_state.gif'
    ani.save(filename, writer='imagemagick', fps=150)
    plt.close()
    print('Done!')
    sys.stdout.flush()

    print('\nGenerating gif for v state...')
    sys.stdout.flush()
    VV = vv.reshape(nt+1, mdl.ny, mdl.nx)
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, lambda i: update_frame(VV, i, '$v(x,y)$'), frames=nt+1, interval=1)
    filename = fom_figs + 'v_state.gif'
    ani.save(filename, writer='imagemagick', fps=150)
    plt.close()
    print('Done!')
    sys.stdout.flush()

# print('Saving data...')
# sys.stdout.flush()
# snapshot_dict = {'parameters': mu_list, 
#                  'snapshots': snapshots}
# pickle.dump(snapshot_dict, open(data_dir1+'snapshots.p', 'wb'))
# print('Done!')
# sys.stdout.flush()
