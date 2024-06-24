import numpy as np
import scipy.sparse as sp

from time import time
from dd_nm_rom.ops import sp_diag
from dd_nm_rom.solvers import Newton


class Burgers2D(object):
  """
  Generate FOM for 2D Burgers equation with Dirichlet BC on the rectangle
  determined by x_lim x y_lim using finite differences.

  inputs:
  nx: number of grid points in x direction
  ny: number of grid points in y direction
  x_lim: x_lim[0] = x-coordinate of left boundary
         x_lim[1] = x-coordinate of right boundary
  y_lim: y_lim[0] = y-coordinate of bottom boundary
         y_lim[1] = y-coordinate of top boundary
  viscosity: positive parameter corresponding to viscosity
  self.u_bc: Dirichlet BC function for u states
  self.v_bc: Dirichlet BC function for v states

  methods:
  set_bc: update boundary condition data
  residual: compute residual of PDE
  res_jac: compute jacobian of residual with respect to [u, v]
  solve: solves for the state u and v using Newton"s method.
  """

  def __init__(
    self,
    nx,
    ny,
    x_lim,
    y_lim,
    viscosity
  ):
    # Grid
    # -------------
    # > Limits
    self.x_lim = x_lim
    self.y_lim = y_lim
    # > Size
    self.nx = nx
    self.ny = ny
    self.nxy = nx*ny
    # > Spacing
    self.hx = (x_lim[1]-x_lim[0])/(nx+1)
    self.hy = (y_lim[1]-y_lim[0])/(ny+1)
    self.hxy = self.hx*self.hy
    # > Points
    self.x = np.linspace(x_lim[0], x_lim[1], nx+2)[1:-1]
    self.y = np.linspace(y_lim[0], y_lim[1], ny+2)[1:-1]
    # PDE paramaters
    # -------------
    self.viscosity = viscosity
    # Differential operators
    # -------------
    self.build_diff_ops()
    # Run time
    # -------------
    self.runtime = 0.0

  def build_diff_ops(self):
    # Vectors/Identities
    ex, Ix = np.ones(self.nx), sp.eye(self.nx)
    ey, Iy = np.ones(self.ny), sp.eye(self.ny)
    # Backward difference matrices
    Bx = sp.spdiags([-ex, ex], [-1, 1], self.nx, self.nx)
    By = sp.spdiags([-ey, ey], [-1, 1], self.ny, self.ny)
    Bx = -0.5/self.hx * sp.kron(Iy, Bx).tocsr()
    By = -0.5/self.hy * sp.kron(By, Ix).tocsr()
    # Centered difference matrices
    Cx = sp.spdiags([ex, -2*ex, ex], [-1, 0, 1], self.nx, self.nx)
    Cy = sp.spdiags([ey, -2*ey, ey], [-1, 0, 1], self.ny, self.ny)
    Cx = self.viscosity/(self.hx*self.hx) * sp.kron(Iy, Cx).tocsr()
    Cy = self.viscosity/(self.hy*self.hy) * sp.kron(Cy, Ix).tocsr()
    # Store matrices
    self.diff_ops = {"Bx": Bx, "By": By, "Cx": Cx, "Cy": Cy}

  def set_bc(self, u_bc, v_bc):
    self.u_bc = u_bc
    self.v_bc = v_bc
    # Vectors
    ex = np.ones(self.nx)
    ey = np.ones(self.ny)
    ex_1_0s = np.concatenate((np.ones(1), np.zeros(self.nx-1)))
    ey_1_0s = np.concatenate((np.ones(1), np.zeros(self.ny-1)))
    ex_0s_1 = np.concatenate((np.zeros(self.nx-1), np.ones(1)))
    ey_0s_1 = np.concatenate((np.zeros(self.ny-1), np.ones(1)))
    # BC
    self.bc = {}
    for z_k in ("u", "v"):
      z_bc = self.u_bc if (z_k == "u") else self.v_bc
      # > Compose
      bc_xl = np.kron(z_bc(self.x_lim[0]*ey, self.y), ex_1_0s)
      bc_xr = np.kron(z_bc(self.x_lim[1]*ey, self.y), ex_0s_1)
      bc_yb = np.kron(ey_1_0s, z_bc(self.x, self.y_lim[0]*ex))
      bc_yt = np.kron(ey_0s_1, z_bc(self.x, self.y_lim[1]*ex))
      # > Compute
      bc_x1 = -(0.5/self.hx)*(bc_xl - bc_xr)
      bc_y1 = -(0.5/self.hy)*(bc_yb - bc_yt)
      bc_x2 = (self.viscosity/(self.hx*self.hx))*(bc_xl + bc_xr)
      bc_y2 = (self.viscosity/(self.hy*self.hy))*(bc_yb + bc_yt)
      # > Store
      self.bc[z_k] = {
        "x": [bc_x1, bc_x2],
        "y": [bc_y1, bc_y2]
      }

  def get_ndof(self):
    return 2*self.nxy

  def rhs(self, x):
    """
    Compute residual of discretized PDE.

    inputs:
    u: (nx*ny,) vector
    v: (nx*ny,) vector

    ouputs:
    res: (2*nx*ny) vector
    """
    dx = []
    uv = {"u": x[:self.nxy], "v": x[self.nxy:]}
    for k in ("u", "v"):
      dx_k = uv["u"]*(self.diff_ops["Bx"] @ uv[k] - self.bc[k]["x"][0]) \
           + uv["v"]*(self.diff_ops["By"] @ uv[k] - self.bc[k]["y"][0]) \
           + self.diff_ops["Cx"] @ uv[k] + self.bc[k]["x"][1] \
           + self.diff_ops["Cy"] @ uv[k] + self.bc[k]["y"][1]
      dx.append(dx_k)
    return np.concatenate(dx)

  def jac(self, x):
    """
    Compute residual jacobian of discretized PDE.

    inputs:
    u: (nx*ny,) vector
    v: (nx*ny,) vector

    ouputs:
    jac: (2*nx*ny, 2*nx*ny) jacobian matrix
    """
    u, v = x[:self.nxy], x[self.nxy:]
    jac_xx = sp_diag(u) @ self.diff_ops["Bx"] \
           + sp_diag(v) @ self.diff_ops["By"] \
           + self.diff_ops["Cx"] + self.diff_ops["Cy"]
    jac_uu = sp_diag(self.diff_ops["Bx"] @ u - self.bc["u"]["x"][0]) + jac_xx
    jac_uv = sp_diag(self.diff_ops["By"] @ u - self.bc["u"]["y"][0])
    jac_vu = sp_diag(self.diff_ops["Bx"] @ v - self.bc["v"]["x"][0])
    jac_vv = sp_diag(self.diff_ops["By"] @ v - self.bc["v"]["y"][0]) + jac_xx
    return sp.bmat(
      [[jac_uu, jac_uv],
       [jac_vu, jac_vv]],
      format="csr"
    )

  def rhs_jac(self, x):
    start = time()
    rhs, jac = self.rhs(x), self.jac(x)
    self.runtime += time()-start
    return rhs, jac

  def solve(
    self,
    x0=None,
    tol=1e-8,
    maxit=50,
    stepsize_min=1e-10,
    verbose=False
  ):
    """
    Solves for the u and v states of the FOM using Newton"s method.

    inputs:
    u0: (nx*ny,) initial u vector
    v0: (nx*ny,) initial v vector
    tol: [optional] stopping tolerance for Newton solver. Default is 1e-10
    maxit: [optional] max number of iterations for newton solver. Default is 100
    print_hist: [optional] Boolean to print iteration history for Newton solver. Default is False

    outputs:
    u: (nx*ny,) u final solution vector
    v: (nx*ny,) v final solution vector
    res_vecs: (it, nx*ny) array where res_vecs[i] is the PDE residual evaluated at the ith Newton iteration
    """
    self.runtime = 0.0
    # Initialize solution
    start = time()
    if (x0 is None):
      x0 = np.zeros(self.get_ndof())
    self.runtime += time()-start
    # Solve
    solver = Newton(
      model=self,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      verbose=verbose
    )
    x, rhs, *_ = solver.solve(x0)
    uv = {"u": x[:self.nxy], "v": x[self.nxy:]}
    return uv, rhs
