# Burgers2D_probgen.py

import numpy as np
import scipy.sparse as sp
from utils.newton_solve import newton_solve

class Burgers2D:
    '''
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
    u_bc: Dirichlet BC function for u states
    v_bc: Dirichlet BC function for v states
    
    methods:
    update_bc: update boundary condition data
    residual: compute residual of PDE
    res_jac: compute jacobian of residual with respect to [u, v]
    solve: solves for the state u and v using Newton's method.
    '''
    def __init__(self, nx, ny, x_lim, y_lim, viscosity, u_bc, v_bc):
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
        
        # quanities used for generating discretization matrices
        ex = np.ones(nx)
        ey = np.ones(ny)
        Ix = sp.eye(nx)
        Iy = sp.eye(ny)
        
        # backward difference matrices
        tBx = sp.spdiags([-ex, ex], [-1, 1], nx, nx)
        tBy = sp.spdiags([-ey, ey], [-1, 1], ny, ny)
        self.Bx  = -(0.5/self.hx)*sp.kron(Iy, tBx).tocsr()
        self.By  = -(0.5/self.hy)*sp.kron(tBy, Ix).tocsr()
        
        # centered difference matrices
        tCx = sp.spdiags([ex, -2*ex, ex], [-1, 0, 1], nx, nx)
        tCy = sp.spdiags([ey, -2*ey, ey], [-1, 0, 1], ny, ny)
        self.Cx  = (self.viscosity/(self.hx*self.hx))*sp.kron(Iy, tCx).tocsr()
        self.Cy  = (self.viscosity/(self.hy*self.hy))*sp.kron(tCy, Ix).tocsr()
        
        self.update_bc(u_bc, v_bc)
    
    def update_bc(self, u_bc, v_bc):
        '''
        Updates boundary condition data 
        
        inputs: 
        u_bc: function for u BC data
        v_bc: function for v BC data
        '''
        # quanities used for generating discretization matrices
        ex = np.ones(self.nx)
        ey = np.ones(self.ny)
        
        # boundary terms for backward difference
        e1_nx = np.concatenate((np.ones(1), np.zeros(self.nx-1)))
        e1_ny = np.concatenate((np.ones(1), np.zeros(self.ny-1)))
        enx_nx = np.concatenate((np.zeros(self.nx-1), np.ones(1)))
        eny_ny = np.concatenate((np.zeros(self.ny-1), np.ones(1)))
        
        b_uxl = np.kron(u_bc(self.x_lim[0]*ey, self.yy), e1_nx)
        b_uxr = np.kron(u_bc(self.x_lim[1]*ey, self.yy), enx_nx)
        b_uyb = np.kron(e1_ny, u_bc(self.xx, self.y_lim[0]*ex))
        b_uyt = np.kron(eny_ny, u_bc(self.xx, self.y_lim[1]*ex))
        self.b_ux1 = -(0.5/self.hx)*(b_uxl - b_uxr)
        self.b_uy1 = -(0.5/self.hy)*(b_uyb - b_uyt)
        self.b_ux2 = (self.viscosity/(self.hx*self.hx))*(b_uxl + b_uxr)
        self.b_uy2 = (self.viscosity/(self.hy*self.hy))*(b_uyb + b_uyt)
        
        b_vxl = np.kron(v_bc(self.x_lim[0]*ey, self.yy), e1_nx)
        b_vxr = np.kron(v_bc(self.x_lim[1]*ey, self.yy), enx_nx)
        b_vyb = np.kron(e1_ny, v_bc(self.xx, self.y_lim[0]*ex)) 
        b_vyt = np.kron(eny_ny, v_bc(self.xx, self.y_lim[1]*ex))
        self.b_vx1 = -(0.5/self.hx)*(b_vxl - b_vxr)
        self.b_vy1 = -(0.5/self.hy)*(b_vyb - b_vyt)
        self.b_vx2 = (self.viscosity/(self.hx*self.hx))*(b_vxl + b_vxr)
        self.b_vy2 = (self.viscosity/(self.hy*self.hy))*(b_vyb + b_vyt)
    
    # compute residual
    def residual(self, u, v):
        '''
        Compute residual of discretized PDE. 
        
        inputs: 
        u: (nx*ny,) vector 
        v: (nx*ny,) vector
        
        ouputs:
        res: (2*nx*ny) vector
        '''
        res_u = u*(self.Bx@u - self.b_ux1) + v*(self.By@u - self.b_uy1) \
              + self.Cx@u + self.b_ux2 + self.Cy@u + self.b_uy2
        res_v = u*(self.Bx@v - self.b_vx1) + v*(self.By@v - self.b_vy1) \
              + self.Cx@v + self.b_vx2 + self.Cy@v + self.b_vy2
        return np.concatenate((res_u, res_v))
    
    # compute Jacobian of the residual
    def res_jac(self, u, v):
        '''
        Compute residual jacobian of discretized PDE. 
        
        inputs: 
        u: (nx*ny,) vector 
        v: (nx*ny,) vector
        
        ouputs:
        jac: (2*nx*ny, 2*nx*ny) jacobian matrix
        '''
        # precompute reused quantities
        diagU_Bx = sp.spdiags(u, 0, self.nxy, self.nxy, format='csr')@self.Bx
        diagV_By = sp.spdiags(v, 0, self.nxy, self.nxy, format='csr')@self.By 
        
        # compute jacobian blocks
        Juu = sp.spdiags(self.Bx@u-self.b_ux1, 0, self.nxy, self.nxy, format='csr') \
              + diagU_Bx + diagV_By + self.Cx + self.Cy 
        Juv = sp.spdiags(self.By@u-self.b_uy1, 0, self.nxy, self.nxy)
        Jvu = sp.spdiags(self.Bx@v-self.b_vx1, 0, self.nxy, self.nxy)
        Jvv = diagU_Bx\
              + sp.spdiags(self.By@v-self.b_vx1, 0, self.nxy, self.nxy, format='csr') \
              + diagV_By + self.Cx + self.Cy 
        return sp.bmat([[Juu, Juv],[Jvu, Jvv]], format='csr')
    
    def solve(self, u0, v0, tol=1e-10, maxit=100, print_hist=False):
        '''
        Solves for the u and v states of the FOM using Newton's method. 
        
        inputs: 
        u0: (nx*ny,) initial u vector
        v0: (nx*ny,) initial v vector
        tol: [optional] stopping tolerance for Newton solver. Default is 1e-10
        maxit: [optional] max number of iterations for newton solver. Default is 100
        print_hist: [optional] Boolean to print iteration history for Newton solver. Default is False
        
        outputs:
        u: (nx*ny,) u final solution vector
        v: (nx*ny,) v final solution vector
        res_vecs: (iter, nx*ny) array where res_vecs[i] is the PDE residual evaluated at the ith Newton iteration
        '''
        
        y0  = np.concatenate([u0, v0])
        F   = lambda y: self.residual(y[0:self.nxy], y[self.nxy:])
        Jac = lambda y: self.res_jac(y[0:self.nxy], y[self.nxy:])
        print('Starting Newton solver...')
        y, res_vecs, res_hist, step_hist, iter = newton_solve(F, Jac, y0, 
                                                              tol=tol, 
                                                              maxit=maxit, 
                                                              print_hist=print_hist, 
                                                              sparse=True)
        print(f'Newton solver terminated after {iter} iterations with residual {res_hist[-1]:1.4e}.')
        return y[0:self.nxy], y[self.nxy:], res_vecs