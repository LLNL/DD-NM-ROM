# domain_decomposition.py
# Author: Alejandro Diaz

import numpy as np
import scipy.sparse as sp
from time import time
from utils.Burgers2D_probgen import Burgers2D
from utils.newton_solve import newton_solve_combinedFJac as newton_solve

class subdomain_indices:
    '''
    Class for generating residual, interior, and interface subdomain indices for a steady-state 2D Burgers FOM.
    
    inputs: 
    fom: instance of Burgers2D class
    num_sub_x: integer number of subdomains in x direction
    num_sub_y: integer number of subdomains in y direction
    
    fields:
    res:       list, res[i] = array of residual indices on subdomain i
    interior:  list, interior[i] = array of interior indices on subdomain i
    interface: list, interface[i] = array of interface indices on subdomain i
    full:      list, full[i] = array of full state (interior and interface) indices on subdomain i
    skeleton:  array of indices of skeleton, i.e. all interface states for all subdomains
    
    '''
    def __init__(self, fom, num_sub_x, num_sub_y):
        n_subs = num_sub_x*num_sub_y
        nx_sub = fom.nx//num_sub_x     # number of x grid points per residual subdomain
        ny_sub = fom.ny//num_sub_y     # number of y grid points per residual subdomain
        full_grid_indices = np.arange(fom.nxy).reshape(fom.ny, fom.nx)

        res_sub_indices   = []   # list of vectors of indices corresponding to each residual subdomain
        full_sub_indices  = []   # list of vectors of indices corresponding to each full state subdomain
        for j in range(num_sub_y):
            for i in range(num_sub_x):
                res_ind = full_grid_indices[ny_sub*j:ny_sub*(j+1), nx_sub*i:nx_sub*(i+1)].flatten()

                full_ind = set(res_ind)
                for row in res_ind:
                    # for each row, finds nonzero columns in fom matrices
                    # leverages coo_matrix format
                    full_ind = full_ind.union(set(fom.Bx.col[fom.Bx.row==row]))    
                    full_ind = full_ind.union(set(fom.By.col[fom.By.row==row]))
                    full_ind = full_ind.union(set(fom.Cx.col[fom.Cx.row==row]))    
                    full_ind = full_ind.union(set(fom.Cy.col[fom.Cy.row==row]))    

                # stores indices for current subdomain
                res_sub_indices.append(np.sort(np.array(res_ind)))
                full_sub_indices.append(full_ind)
        
        interior_indices  = []   # list of vectors of indices corresponding to each interior state subdomain
        interface_indices = []   # list of vectors of indices corresponding to each interface state subdomain
        skeleton = set([])       # list of all interface nodes
            
        for i in range(n_subs):
            subdomain_i = full_sub_indices[i]
            
            interface = set([])
            other_subdomains = np.concatenate([np.arange(0, i), np.arange(i+1,n_subs)])
            
            # takes union of intersections of full subdomain indices of subdomains i and j, i=/=j
            for j in other_subdomains:
                subdomain_j   = full_sub_indices[j]
                i_intersect_j = subdomain_i.intersection(subdomain_j)
                
                # if nonempty intersection, add to interface indices
                if len(i_intersect_j) > 0:
                    interface = interface.union(i_intersect_j)
            
            # finds interior indices as set difference of full subdomain and interface indices
            interior = subdomain_i.difference(interface)
            skeleton = skeleton.union(interface)
            interface_indices.append(np.sort(np.array(list(interface))))
            interior_indices.append(np.sort(np.array(list(interior))))
                
        self.res = res_sub_indices
        self.interior = interior_indices
        self.interface = interface_indices
        self.skeleton = np.array(list(skeleton))
        self.full = [np.sort(np.array(list(ind))) for ind in full_sub_indices]
        
class subdomain_model:
    '''
    Class for generating a subdomain of the DD-FOM for the 2D steady-state Burgers' Equation with Dirichlet BC.
    
    inputs: 
    fom: instance of Burgers2D class representing full domain problem 
    residual_ind: array of residual indices corresponding to the subdomain to be generated
    interior_ind: array of interior indices corresponding to the subdomain to be generated
    interface_ind: array of interface_ind indices corresponding to the subdomain to be generated
    constraint_mat: constraint matrix corresponding to the interface states of the subdomain
    in_ports: array containing which ports the subdomain belongs toa
    
    methods:
    update_bc: update boundary condition data on subdomain
    res_jac: compute residual and its jacobian on the subdomain
    '''
    def __init__(self, fom, residual_ind, interior_ind, interface_ind, constraint_mat, in_ports):
        # stores necessary fom quantities
        self.hx = fom.hx
        self.hy = fom.hy
        self.viscosity = fom.viscosity
        self.constraint_mat = constraint_mat
        self.in_ports = in_ports
        
        self.n_residual = len(residual_ind)
        self.n_interior = len(interior_ind)
        self.n_interface = len(interface_ind)
        
        self.residual_ind  = residual_ind     # indices on residual subdomain
        self.interior_ind  = interior_ind     # indices on subdomain interior
        self.interface_ind = interface_ind    # indices on subdomain interface
        
        # finds where indices in interior/interface are in the residual indices
        self.interior2res  = np.nonzero(np.isin(interior_ind, residual_ind))[0]
        self.res2interior  = np.nonzero(np.isin(residual_ind, interior_ind))[0]
        self.interface2res = np.nonzero(np.isin(interface_ind, residual_ind))[0]
        self.res2interface = np.nonzero(np.isin(residual_ind, interface_ind))[0]
        
        # inclusion operator for interior/interface states to residual subdomain
        self.I_interior = sp.csr_matrix((np.ones(len(self.interior2res)), (self.res2interior, self.interior2res)), 
                                         shape=(self.n_residual, self.n_interior))
        self.I_interface = sp.csr_matrix((np.ones(len(self.interface2res)), (self.res2interface, self.interface2res)), 
                                         shape=(self.n_residual, self.n_interface))
        
        Bx = fom.Bx.tocsr()
        By = fom.By.tocsr()
        Cx = fom.Cx.tocsr()
        Cy = fom.Cy.tocsr()
        
        # finds submatrices for interior state
        self.Bx_interior  = Bx[np.ix_(residual_ind, interior_ind)]
        self.By_interior  = By[np.ix_(residual_ind, interior_ind)]
        self.Cx_interior  = Cx[np.ix_(residual_ind, interior_ind)]
        self.Cy_interior  = Cy[np.ix_(residual_ind, interior_ind)]
        
        # finds submatrices for interface states
        self.Bx_interface = Bx[np.ix_(residual_ind, interface_ind)]
        self.By_interface = By[np.ix_(residual_ind, interface_ind)]
        self.Cx_interface = Cx[np.ix_(residual_ind, interface_ind)]
        self.Cy_interface = Cy[np.ix_(residual_ind, interface_ind)]
        
        # BC data for subdomain
        self.update_bc(fom)
    
    # updates subdomain BC data
    def update_bc(self, fom):
        '''
        Updates boundary condition data on current subdomain
        
        inputs: 
        fom: instance of Burgers2D class with updated BC data
        '''
        self.b_ux1 = fom.b_ux1[self.residual_ind]
        self.b_uy1 = fom.b_uy1[self.residual_ind]
        self.b_ux2 = fom.b_ux2[self.residual_ind]
        self.b_uy2 = fom.b_uy2[self.residual_ind]
        
        self.b_vx1 = fom.b_vx1[self.residual_ind]
        self.b_vy1 = fom.b_vy1[self.residual_ind]
        self.b_vx2 = fom.b_vx2[self.residual_ind]
        self.b_vy2 = fom.b_vy2[self.residual_ind]
    
    # computes subdomain residual
    def res_jac(self, u_interior, v_interior, u_interface, v_interface, lam):
        '''
        Compute residual and its jacobians with respect to interior and interface states. 
        
        inputs: 
        u_interior: (n_interior,) vector of u interior states
        v_interior: (n_interior,) vector of v interior states
        u_interface: (n_interface,) vector of u interface states
        v_interface: (n_interface,) vector of  v interface states
        lam        : (n_constraints,) vector of lagrange multipliers
        
        outputs:
        res: (2*n_residual,) residual vector with u and v residuals concatenated
        jac: (2*n_residual, n_interior+n_interface) array - jacobian of residual w.r.t. (w_intr, w_intf)
        H:   Hessian submatrix for SQP solver
        rhs: RHS block vector in SQP solver
        Ax : constraint matrix times interface state
        
        '''
        # assemble u and v on residual subdomains
        u_res = np.zeros(len(self.residual_ind))
        u_res[self.res2interior]  = u_interior[self.interior2res]
        u_res[self.res2interface] = u_interface[self.interface2res]
        
        v_res = np.zeros(len(self.residual_ind))
        v_res[self.res2interior]  = v_interior[self.interior2res]
        v_res[self.res2interface] = v_interface[self.interface2res]
        
        # compute matrix-vector products on interior and interface
        Bxu = self.Bx_interior@u_interior + self.Bx_interface@u_interface
        Byu = self.By_interior@u_interior + self.By_interface@u_interface
        Cxu = self.Cx_interior@u_interior + self.Cx_interface@u_interface
        Cyu = self.Cy_interior@u_interior + self.Cy_interface@u_interface
        
        Bxv = self.Bx_interior@v_interior + self.Bx_interface@v_interface
        Byv = self.By_interior@v_interior + self.By_interface@v_interface
        Cxv = self.Cx_interior@v_interior + self.Cx_interface@v_interface
        Cyv = self.Cy_interior@v_interior + self.Cy_interface@v_interface
        
        # computes u and v residuals
        res_u = u_res*(Bxu - self.b_ux1) + v_res*(Byu - self.b_uy1) \
              + Cxu + self.b_ux2 + Cyu + self.b_uy2
        res_v = u_res*(Bxv - self.b_vx1) + v_res*(Byv - self.b_vy1) \
              + Cxv + self.b_vx2 + Cyv + self.b_vy2
        res = np.concatenate((res_u, res_v))
        
        # precompute reused quantities
        diagU = sp.spdiags(u_res, 0, self.n_residual, self.n_residual)
        diagV = sp.spdiags(v_res, 0, self.n_residual, self.n_residual)
        diagU_Bx_interior = diagU@self.Bx_interior
        diagV_By_interior = diagV@self.By_interior
        diagU_Bx_interface = diagU@self.Bx_interface
        diagV_By_interface = diagV@self.By_interface
        
        # jacobian with respect to interior states
        jac_interior_uu = sp.spdiags(Bxu-self.b_ux1, 0, self.n_residual, self.n_residual)@self.I_interior \
                          + diagU_Bx_interior + diagV_By_interior + self.Cx_interior + self.Cy_interior 
        jac_interior_uv = sp.spdiags(Byu-self.b_uy1, 0, self.n_residual, self.n_residual)@self.I_interior
        jac_interior_vu = sp.spdiags(Bxv-self.b_vx1, 0, self.n_residual, self.n_residual)@self.I_interior
        jac_interior_vv = diagU_Bx_interior \
                          + sp.spdiags(Byv-self.b_vx1, 0, self.n_residual, self.n_residual)@self.I_interior \
                          + diagV_By_interior + self.Cx_interior + self.Cy_interior
        jac_interior = sp.bmat([[jac_interior_uu, jac_interior_uv],[jac_interior_vu, jac_interior_vv]], format='csr')
        
        # jacobian with respect to interface states
        jac_interface_uu = sp.spdiags(Bxu-self.b_ux1, 0, self.n_residual, self.n_residual)@self.I_interface \
                          + diagU_Bx_interface + diagV_By_interface + self.Cx_interface + self.Cy_interface
        jac_interface_uv = sp.spdiags(Byu-self.b_uy1, 0, self.n_residual, self.n_residual)@self.I_interface
        jac_interface_vu = sp.spdiags(Bxv-self.b_vx1, 0, self.n_residual, self.n_residual)@self.I_interface
        jac_interface_vv = diagU_Bx_interface + sp.spdiags(Byv-self.b_vx1, 0, self.n_residual, self.n_residual)@self.I_interface \
                          + diagV_By_interface + self.Cx_interface + self.Cy_interface
        jac_interface = sp.bmat([[jac_interface_uu, jac_interface_uv],[jac_interface_vu, jac_interface_vv]], format='csr')
        
        # compute terms needed for SQP solver
        Ax = self.constraint_mat@np.concatenate([u_interface, v_interface])
        
        jac = sp.hstack([jac_interior, jac_interface])
        H   = jac.T@jac
        rhs = np.concatenate([jac_interior.T@res, jac_interface.T@res+self.constraint_mat.T@lam])
        
        return res, jac, H, rhs, Ax
    
class DD_model:
    '''
    Class to compute domain decomposition model from steady-state 2D Burgers FOM
    
    inputs:
    fom: instance of Burgers2D class
    num_sub_x: number of subdomains in x direction. Must divide fom.nx
    num_sub_y: number of subdmoains in y direction. Must divide fom.ny
    
    fields:
    nxy:       total number of nodes (finite difference grid points) in full domain model
    n_sub:     number of subdomains
    skeleton:  indices of each node in the skeleton
    ports:     list of frozensets corresponding to each port in the DD model. 
                  ports[i] = frozenset of the subdomains contained in port i
    port_dict: dictionary of port indices where 
                  port_dict[port[i]] = indices in port[i]
    n_constraints: number of (equality) constraints for DD model
    subdomain: list of instances of subdomain_model class where
                  subdomain[i] = subdomain_model instance corresponding to ith subdomain
    
    methods:
    update_bc: update boundary condition data
    FJac: computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver
    solve: solves for the states of the DD model using the Lagrange-Newton-SQP method
    '''
    def __init__(self, fom, num_sub_x, num_sub_y,
                 constraint_type='strong', n_constraints=1, seed=None):
        self.nxy = fom.nxy
        self.n_sub = num_sub_x*num_sub_y
        
        # generate subdomain indices for given number of subdomains
        indices = subdomain_indices(fom, num_sub_x, num_sub_y)
        
        # sort each interface node into a port
        port_array = np.zeros((len(indices.skeleton), self.n_sub), dtype=bool)
        for i in range(indices.skeleton.size):
            for j in range(self.n_sub):
                port_array[i, j] = indices.skeleton[i] in indices.interface[j]
        
        # extract ports from port_array
        ports = set([])
        sub_list = np.arange(self.n_sub)
        for row in port_array:
            ports.add(frozenset(sub_list[row]))
        ports = list(ports)
        
        # dictionary for storing nodes in each port
        port_dict = {}
        for port in ports:
            ind = np.zeros(self.n_sub, dtype=bool)
            ind[np.array(list(port))] = True
            port_dict[port]=np.sort(indices.skeleton[(port_array == list(ind)).all(axis=1)])
            
        self.skeleton  = indices.skeleton      # list of each node in skeleton
        self.ports     = ports         # list of each port
        self.port_dict = port_dict     # dictionary where port_dict[port] = list of nodes in port
        
        # find which ports each subdomain belongs to
        in_ports = []
        for i in range(self.n_sub):
            in_ports_i = set([])
            for j in range(len(self.ports)):
                if i in list(self.ports[j]):
                    in_ports_i.add(j)
            in_ports.append(np.sort(list(in_ports_i)))
    
        # generate port constraint matrices
        self.n_constraints = np.sum([(len(port)-1)*len(port_dict[port]) for port in ports])
        constraint_mat = [sp.coo_matrix((self.n_constraints, len(ind))) for ind in indices.interface]
        shift = 0 
        for j in range(len(ports)):
            port = list(ports[j])
            port_indices = port_dict[ports[j]]
            npj = len(port_indices)
            for i in range(len(port)-1):
                p1   = port[i]
                ind1 = np.where(np.isin(indices.interface[p1], port_indices))[0]
                constraint_mat[p1].col  = np.concatenate((constraint_mat[p1].col, ind1))
                constraint_mat[p1].row  = np.concatenate((constraint_mat[p1].row, np.arange(0, npj)+shift))
                constraint_mat[p1].data = np.concatenate((constraint_mat[p1].data, np.ones(npj)))   

                p2   = port[i+1]
                ind2 = np.where(np.isin(indices.interface[p2], port_indices))[0]
                constraint_mat[p2].col  = np.concatenate((constraint_mat[p2].col, ind2))
                constraint_mat[p2].row  = np.concatenate((constraint_mat[p2].row, np.arange(0, npj)+shift))
                constraint_mat[p2].data = np.concatenate((constraint_mat[p2].data, -np.ones(npj)))

                shift += npj
        
        # make constraint matrix block diagonal for u, v components
        constraint_mat = [sp.block_diag([A, A]) for A in constraint_mat]
        self.n_constraints *= 2
        
        # transform to weak constraints
        if constraint_type == 'weak':
            rng = np.random.default_rng(seed)
            constraint_mult = rng.standard_normal(size=(n_constraints, self.n_constraints))
            self.n_constraints = n_constraints
            constraint_mat = [constraint_mult@A for A in constraint_mat]
            
        # generates model for each subdomain
        self.subdomain = list([])
        for i in range(self.n_sub):
            self.subdomain.append(subdomain_model(fom, 
                                                  indices.res[i], 
                                                  indices.interior[i],
                                                  indices.interface[i], 
                                                  constraint_mat[i], 
                                                  in_ports[i]))
            
    # update BC data for each subdomain
    def update_bc(self, fom):
        '''
        Updates boundary condition data on current subdomain
        
        inputs: 
        fom: instance of Burgers2D class with updated BC data
        '''
        for s in self.subdomain:
            s.update_bc(fom)
    
    def FJac(self, w):
        '''
        Computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver. 
        
        inputs: 
        w: vector of all interior and interface states for each subdomain 
           and the lagrange multipliers lam in the order
                    w = [u_intr[0], 
                         v_intr[0],
                         u_intf[0], 
                         v_intf[0],
                         ..., 
                         u_intr[n_sub], 
                         v_intr[n_sub], 
                         u_intf[n_sub], 
                         v_intf[n_sub],
                         lam]
            
        outputs:
        val: RHS of the KKT system
        full_jac: KKT matrix
        
        '''
        shift = 0
        val = list([])
        A_list = list([])
        H_list = list([])

        constraint_res = np.zeros(self.n_constraints)
        lam = w[-self.n_constraints:]
        for s in self.subdomain:
            interior_ind  = np.arange(s.n_interior)
            interface_ind = np.arange(s.n_interface)

            u_interior = w[interior_ind+shift]
            shift     += s.n_interior
            v_interior = w[interior_ind+shift]
            shift     += s.n_interior

            u_interface = w[interface_ind+shift]
            shift      += s.n_interface
            v_interface = w[interface_ind+shift]
            shift      += s.n_interface
            
            # computes residual, jacobian, and other quantities needed for KKT system
            res, jac, H, rhs, Ax = s.res_jac(u_interior, v_interior, u_interface, v_interface, lam)
        
            # RHS block for KKT system
            val.append(rhs)
            constraint_res += Ax
            
            H_list.append(H)
            A_list += [sp.csr_matrix((self.n_constraints, 2*s.n_interior)), s.constraint_mat]
            
#             res, jac_interior, jac_interface = s.res_jac(u_interior, v_interior, u_interface, v_interface)
            
#             ATlam = s.constraint_mat.T@lam

#             val.append(np.concatenate([jac_interior.T@res, jac_interface.T@res+ATlam]))
            
#             constraint_res += s.constraint_mat@np.concatenate([u_interface, v_interface])

#             jac = sp.hstack([jac_interior, jac_interface])
#             H = jac.T@jac
            
#             H_list.append(H)
#             A_list += [sp.csr_matrix((self.n_constraints, 2*s.n_interior)), s.constraint_mat]

        val.append(constraint_res)
        val = np.concatenate(val)
        H_block = sp.block_diag(H_list)
        A_block = sp.hstack(A_list)
        full_jac = sp.bmat([[H_block, A_block.T], [A_block, None]]).tocsr()
        return val, full_jac
    
    def solve(self, w0, tol=1e-5, maxit=20, print_hist=False):
        '''
        Solves for the u and v interior and interface states of the DD-FOM using the Lagrange-Newton-SQP algorithm. 
        
        inputs: 
        w0: initial interior/interface states and lagrange multipliers
        tol: [optional] stopping tolerance for Newton solver. Default is 1e-5
        maxit: [optional] max number of iterations for newton solver. Default is 20
        print_hist: [optional] Boolean to print iteration history for Newton solver. Default is False
        
        outputs:
        u_full: u state mapped to full domain
        v_full: v state mapped to full domain
        u_interior: list of u interior states for each subdomain, i.e
                    u_interior[i] = u interior state on subdomain i
        v_interior: list of v interior states for each subdomain, i.e
                    v_interior[i] = v interior state on subdomain i
        u_interface: list of u interface states for each subdomain, i.e
                     u_interface[i] = u interface state on subdomain i  
        v_interface: list of v interface states for each subdomain, i.e
                     v_interface[i] = v interface state on subdomain i
        lam:        vector of optimal Lagrange multipliers
        runtime: solve time for Newton solver
        itr: number of iterations for Newton solver
        '''
        print('Starting Newton solver...')
        start = time()
        y, res_vecs, res_hist, step_hist, itr = newton_solve(self.FJac, w0, tol=tol, maxit=maxit, print_hist=print_hist)
        runtime = time()-start
        print(f'Newton solver terminated after {itr} iterations with residual {res_hist[-1]:1.4e}.')
        
        # assemble solution on full domain from DD solution
        u_full = np.zeros(self.nxy)
        v_full = np.zeros(self.nxy)
        u_interior = list([])
        v_interior = list([])
        u_interface = list([])
        v_interface = list([])
        lam = y[-self.n_constraints:]
        
        shift = 0
        for s in self.subdomain:
            u_full[s.interior_ind] = y[np.arange(s.n_interior)+shift]
            u_interior.append(y[np.arange(s.n_interior)+shift])
            shift += s.n_interior 
            
            v_full[s.interior_ind] = y[np.arange(s.n_interior)+shift]
            v_interior.append(y[np.arange(s.n_interior)+shift])
            shift += s.n_interior 

            u_full[s.interface_ind] = y[np.arange(s.n_interface)+shift]
            u_interface.append(y[np.arange(s.n_interface)+shift])
            shift += s.n_interface    
            
            v_full[s.interface_ind] = y[np.arange(s.n_interface)+shift]
            v_interface.append(y[np.arange(s.n_interface)+shift])
            shift += s.n_interface
            
        return u_full, v_full, u_interior, v_interior, u_interface, v_interface, lam, runtime, itr