'''
Generate snapshot data for the 2D Burgers
equation for a number of input parameters.
'''

import sys
with open("./../../PATHS.txt") as file:
  paths = file.read().splitlines()
sys.path.extend(paths)

import os
import argparse
import numpy as np
import dill as pickle

from dd_nm_rom.fom import Burgers2D, Burgers2DExact


# Inputs parser
# =====================================
parser = argparse.ArgumentParser()
parser.add_argument(
  "--nx", default=240, type=int,
  help="number of grid points in x-direction"
)
parser.add_argument(
  "--ny", default=12, type=int,
  help="number of grid points in y-direction"
)
parser.add_argument(
  "--na", default=20, type=int,
  help="number of grid points in a-direction on parameter grid"
)
parser.add_argument(
  "--nlam", default=80, type=int,
  help="number of grid points in lambda-direction on parameter grid"
)
parser.add_argument(
  "--viscosity", default=1e-1, type=float,
  help="viscosity for 2D Burgers' problem"
)
parser.add_argument(
  "--maxit", default=80, type=int,
  help="maximum number of iterations for Newton solver"
)
parser.add_argument(
  "--tol", default=1e-8, type=float,
  help="absolute tolerance for Newton solver"
)
parser.add_argument(
  "--save_res", default=1, type=int, choices=[0,1],
  help="save residual snapshots"
)
parser.add_argument(
  "--save_dir", default='./data/', type=str,
  help="path to saving folder"
)
args = parser.parse_args()

# Set up PDE
# =====================================
# PDE discretization parameters
x_lim = [-1.0, 1.0]
y_lim = [0.0, 0.05]
nx, ny = args.nx, args.ny
viscosity = args.viscosity

# Parameter space
a1_lim = [1.0, 10000.0]
lam_lim = [5.0, 25.0]
na, nlam = args.na, args.nlam

print('\nComputing state snapshots with the following parameters:')
print(f'nx        = {nx}')
print(f'ny        = {ny}')
print(f'viscosity = {viscosity}')
print(f'na        = {na}')
print(f'nlam      = {nlam}')

print('\nParameters for Newton solver:')
print(f'maxit     = {args.maxit}')
print(f'tol       = {args.tol}')
sys.stdout.flush()

# Generate samples
# =====================================
# Generates parameters
a1_vals = np.linspace(a1_lim[0], a1_lim[1], na)
lam_vals = np.linspace(lam_lim[0], lam_lim[1], nlam)
A1, Lam = np.meshgrid(a1_vals, lam_vals)
Mu = np.vstack([A1.flatten(), Lam.flatten()]).T

fom_ex = Burgers2DExact()
fom = Burgers2D(nx, ny, x_lim, y_lim, viscosity)

# compute solution for each pair of parameters
u0 = np.zeros(nx*ny)
v0 = np.zeros(nx*ny)
snapshots, residuals = [], []
for mu in Mu:
  # sets current parameters
  a1, lam = mu
  print(f'\n> Computing solutions for (a1,lam) = ({a1}, {lam}) ...')

  fom_ex.set_params(a1, lam, viscosity)
  fom.set_bc(fom_ex.u, fom_ex.v)
  uv, rhs = fom.solve(tol=args.tol, maxit=args.maxit, stepsize_min=1e-20, verbose=False)
  sol = np.concatenate([uv["u"], uv["v"]])

  # store snapshots and intermediate Newton residuals
  snapshots.append(sol)
  residuals.append(rhs)
  sys.stdout.flush()

# save data to file
save_dict = {
  'parameters': Mu,
  'snapshots': snapshots
}
if args.save_res:
  save_dict['residuals'] = residuals

os.makedirs(args.save_dir, exist_ok=True)
file = args.save_dir + f'/nx{nx}_ny{ny}_mu{viscosity}_nsamples{len(Mu)}.p'
pickle.dump(save_dict, open(file,'wb'))
