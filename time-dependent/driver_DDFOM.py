import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from utils.Burgers2D_probgen import Burgers2D
from utils.domain_decomposition import DD_FOM
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
parser.add_argument("--tol", default=1e-9, type=float, 
                    help="absolute tolerance for Newton solver")
parser.add_argument("--mu", default=1.0, type=float, 
                    help="Amplitude of initial condition (parameter for PDE)")
args = parser.parse_args()

# parameters for physical domain and FD discretization
x_lim = [0, 1]
y_lim = [0, 1]
t_lim = [0, 2]
nx, ny, nt = args.nx, args.ny, args.nt
viscosity = args.viscosity
nsub_x, nsub_y = args.nsub_x, args.nsub_y
mu = args.mu

print('\nComputing state snapshots with the following parameters:')
print(f'nsub_x    = {nsub_x}')
print(f'nsub_y    = {nsub_y}')
print(f'nx        = {nx}')
print(f'ny        = {ny}')
print(f'nt        = {nt}')
print(f'viscosity = {viscosity}')
print(f'mu        = {mu}')

print('\nParameters for Newton solver:')
print(f'maxit = {args.maxit}')
print(f'tol   = {args.tol}')
sys.stdout.flush()

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
def update_frame(Z, i, cb_label):
        plt.clf()
        plt.pcolormesh(X, Y, Z[i], cmap='viridis', shading='auto', vmin=Z.min(), vmax=Z.max()) 
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        cb = plt.colorbar(orientation='vertical', label=cb_label)
        return plt

print(f'\nRunning for mu = {mu}:')
sys.stdout.flush()

# make figure directories
fig_dir2 = fig_dir1 + f'mu_{mu}/'
fom_figs = fig_dir2 + f'ddfom_{nsub_x}x_by_{nsub_y}y/'
for d in [fig_dir2, fom_figs]:
    if not os.path.exists(d): os.mkdir(d)

# set initial condition
ddfom.set_initial(lambda xy: u0(xy, mu), lambda xy: v0(xy, mu))

# solve Burgers equation
print('\nSolve DD Burgers equation:')
sys.stdout.flush()
u_full, v_full, u_intr, v_intr, u_intf, v_intf, lam, runtime, flag = ddfom.solve(t_lim,
                                                                           nt, 
                                                                           tol=args.tol,
                                                                           maxit=args.maxit)
print('Done!') 
sys.stdout.flush()

# store snapshots
print('Saving data...')
sys.stdout.flush()
for i in range(ddfom.n_sub):
    sub_dir  = data_dir2 + f'sub_{i+1}of{ddfom.n_sub}/'
    intr_dir = sub_dir + 'interior/'
    intf_dir = sub_dir + 'interface/'
    for d in [sub_dir, intr_dir, intf_dir]:
        if not os.path.exists(d): os.mkdir(d)
    uv_intr = {'solution': np.hstack([u_intr[i], v_intr[i]]), 'runtime': runtime}
    uv_intf = {'solution': np.hstack([u_intf[i], v_intf[i]]), 'runtime': runtime}
    pickle.dump(uv_intr, open(intr_dir+f'mu_{mu}_uv_state.p', 'wb'))
    pickle.dump(uv_intf, open(intf_dir+f'mu_{mu}_uv_state.p', 'wb'))
print('Done!')
print(f'Runtime = {runtime:1.5e} seconds')
sys.stdout.flush()

# save gifs of solutions
print('\nGenerating gif for u state...')
sys.stdout.flush()
UU = u_full.reshape(nt+1, ddfom.ny, ddfom.nx)
fig = plt.figure()
ani = animation.FuncAnimation(fig, lambda i: update_frame(UU, i, '$u(x,y)$'), frames=nt+1, interval=1)
filename = fom_figs + 'u_state.gif'
ani.save(filename, writer='imagemagick', fps=nt//10)
plt.close()
print('Done!')
sys.stdout.flush()

print('\nGenerating gif for v state...')
sys.stdout.flush()
VV = v_full.reshape(nt+1, ddfom.ny, ddfom.nx)
fig = plt.figure()
ani = animation.FuncAnimation(fig, lambda i: update_frame(VV, i, '$v(x,y)$'), frames=nt+1, interval=1)
filename = fom_figs + 'v_state.gif'
ani.save(filename, writer='imagemagick', fps=nt//10)
plt.close()
print('Done!')
sys.stdout.flush()