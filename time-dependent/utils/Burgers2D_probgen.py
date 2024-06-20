import numpy as np
import scipy.sparse as sp 
from utils.helpers import sp_diag
from utils.solvers import newton_solve
from time import time
import sys

class Burgers2D:
    '''
    Generate FOM for 2D Burgers equation with homogeneous Dirichlet BC on the rectangle
    determined by x_lim x y_lim using finite differences. 
    
    inputs: 
    nx: number of grid points in x direction
    ny: number of grid points in y direction
    nt: number of time steps
    x_lim: x_lim[0] = x-coordinate of left boundary
           x_lim[1] = x-coordinate of right boundary
    y_lim: y_lim[0] = y-coordinate of bottom boundary
           y_lim[1] = y-coordinate of top boundary
    viscosity: positive parameter corresponding to viscosity
    u0: Initial condition function for u states
    v0: Initial condition function for v states
    
    methods:
    set_initial: update initial condition data
    res_jac: compute jacobian of residual with respect to [u, v]
    solve: solves for the state u and v using Newton's method.
    '''
    def __init__(self, nx, ny, x_lim, y_lim, viscosity): #, u0, v0):
        # grid spacing
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.nx  = nx
        self.ny  = ny
        self.nxy = nx*ny
        
        self.hx = (x_lim[1]-x_lim[0])/(nx+1)
        self.hy = (y_lim[1]-y_lim[0])/(ny+1)
        self.hxy = self.hx*self.hy
        self.viscosity = viscosity
               
        # x and y grid points
        self.xx = np.linspace(x_lim[0], x_lim[1], nx+2)[1:-1]
        self.yy = np.linspace(y_lim[0], y_lim[1], ny+2)[1:-1]
        X, Y = np.meshgrid(self.xx, self.yy)
        self.XY = np.vstack([X.reshape(-1,), Y.reshape(-1,)]).T
        
        # quanities used for generating discretization matrices
        ex = np.ones(nx)
        ey = np.ones(ny)
        Ix = sp.eye(nx)
        Iy = sp.eye(ny)
        self.Ixy = sp.eye(self.nxy).tocsr()
        
        # backward difference matrices
        tBx = sp.spdiags([-ex, ex], [-1, 0], nx, nx)
        tBy = sp.spdiags([-ey, ey], [-1, 0], ny, ny)
        self.Bx  = -(1.0/self.hx)*sp.kron(Iy, tBx).tocsr()
        self.By  = -(1.0/self.hy)*sp.kron(tBy, Ix).tocsr()
        
        # centered difference matrices
        tCx = sp.spdiags([ex, -2*ex, ex], [-1, 0, 1], nx, nx)
        tCy = sp.spdiags([ey, -2*ey, ey], [-1, 0, 1], ny, ny)
        self.C = (self.viscosity/(self.hx*self.hx))*sp.kron(Iy, tCx).tocsr()\
                 +(self.viscosity/(self.hy*self.hy))*sp.kron(tCy, Ix).tocsr()
    
#         self.set_initial(u0, v0)
    
    def set_initial(self, u0, v0):
        '''
        Update initial condition.
        
        inputs:
        u0:  function for initial u
        v0:  function for initial v
        '''
        self.u0 = u0(self.XY)
        self.v0 = v0(self.XY)
    
    # compute residual
    def res_jac(self, un, vn, uc, vc, ht):
        '''
        Compute residual and jacobian of discretized PDE. 
        
        inputs: 
        un:  (nx*ny,) vector, u at next time step
        vn:  (nx*ny,) vector, v at next time step
        uc:  (nx*ny,) vector, u at current time step
        vc:  (nx*ny,) vector, v at current time step
        ht:  time step
        
        ouputs:
        res: (2*nx*ny) vector
        jac: (2*nx*ny, 2*nx*ny) jacobian matrix
        '''
        start = time()
        Bxu = self.Bx@un
        Byu = self.By@un
        Bxv = self.Bx@vn
        Byv = self.By@vn
        UBx_VBy_C = sp_diag(un)@self.Bx + sp_diag(vn)@self.By + self.C
        
        res_u = un - uc - ht*(un*Bxu + vn*Byu + self.C@un)
        res_v = vn - vc - ht*(un*Bxv + vn*Byv + self.C@vn)
        
        Juu = self.Ixy - ht*(sp_diag(Bxu) + UBx_VBy_C)
        Juv = -ht*sp_diag(Byu)
        Jvu = -ht*sp_diag(Bxv)
        Jvv = self.Ixy - ht*(sp_diag(Byv) + UBx_VBy_C)
        
        return np.concatenate([res_u, res_v]), sp.bmat([[Juu, Juv], [Jvu, Jvv]], format='csr'), time()-start
    
    def solve(self, t_lim, nt, tol=1e-9, maxit=20, print_hist=False):
        '''
        Solve 2D Burgers' IVP. 
        
        inputs:
        t_lim: t_lim[0] = initial time
               t_lim[1] = final time 
        nt:    number of time steps
        tol:        [optional] solver relative tolerance. Default is 1e-10
        maxit:      [optional] maximum number of iterations. Default is 20
        print_hist: [optional] Set to True to print iteration history. Default is False
        
        outputs:
        uu:         (nt+1, nx*ny) array. u component of Burgers' equation solution
        vv:         (nt+1, nx*ny) array. v component of Burgers' equation solution
        res_hist:   (nt+1) list of Newton iterate residual for each time step
        step_hist:  (nt+1) list of Newton step sizes for each time step
        runtime:    wall clock time for solve
        
        '''
        start=time()
        ht = (t_lim[1]-t_lim[0])/nt
        uu = np.zeros((nt+1, self.nxy))
        vv = np.zeros((nt+1, self.nxy))
        uu[0] = self.u0
        vv[0] = self.v0
        
        uc = self.u0
        vc = self.v0
        
        res_hist  = []
        step_hist = []
        runtime=time()-start
        
        for k in range(nt):
            start = time()
            if print_hist: print(f'Time step {k}:')
            FJac = lambda w: self.res_jac(w[:self.nxy], w[self.nxy:], uc, vc, ht)
            runtime += time()-start
            
            y, rh, norm_res, sh, iter, rtk, flag = newton_solve(FJac, 
                                                          np.concatenate([uc, vc]), 
                                                          tol=tol, 
                                                          maxit=maxit, 
                                                          print_hist=print_hist)
                
            runtime += rtk
            
            start=time()
            res_hist.append(rh)
            step_hist.append(sh)
            uc, vc = y[:self.nxy], y[self.nxy:]
            uu[k+1] = uc
            vv[k+1] = vc
            if flag == 1: 
                print(f'Time step {k+1}: solver failed to converge in {maxit} iterations.')
                sys.stdout.flush()
                break

            elif flag == 2:
                print(f'Time step {k+1}: no stepsize found.')
                sys.stdout.flush()
                break
            runtime += time()-start
            
            
        return uu, vv, res_hist, step_hist, runtime, flag