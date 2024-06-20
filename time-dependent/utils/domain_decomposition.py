import numpy as np
import scipy.sparse as sp
from time import time
from utils.solvers import newton_solve
from utils.helpers import sp_diag
import sys
import dill as pickle

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
                    for M in [fom.Bx, fom.By, fom.C]:
                        M = M.tocoo()
                        full_ind = full_ind.union(set(M.col[M.row==row]))  

                # stores indices for current subdomain
                res_sub_indices.append(np.sort(np.array(res_ind)))
                full_sub_indices.append(full_ind)
        
        interior_indices  = []   # list of vectors of indices corresponding to each interior state subdomain
        interface_indices = []   # list of vectors of indices corresponding to each interface state subdomain
        skeleton = set([])       # list of all interface nodes
            
        for i in range(n_subs):
            subdomain_i = full_sub_indices[i]
            
            interface = set([])
            other_subdomains = [j for j in range(n_subs) if j != i]
            
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
                
        self.residual = res_sub_indices
        self.interior = interior_indices
        self.interface = interface_indices
        self.skeleton = np.array(list(skeleton))
        self.full = [np.sort(np.array(list(ind))) for ind in full_sub_indices]
        
class state_component:
    '''
    Generate FOM state component from indices.
    '''
    def __init__(self, fom, residual_ind, state_ind):
        '''
        Instantiate state_component class. 
        
        inputs:
        fom:            Burgers2D class
        residual_ind:   vector of indices corresponding to subdomain residual
        state_ind:      vector of indices corresponding to subdomain state (interior or interface)
        
        fields: 
        indices:        indices of monolithic FOM corresponding to state component
        size:           number of nodes in state component
        Bx:             Bx matrix for state component
        Bx:             Bx matrix for state component
        C:              C matrix for state component
        I:              residual inclusion matrix for state component
        XY:             FOM spatial gridpoints corresponding to state component
        
        methods:
        set_initial:    set initial condition for state component
        '''
        self.indices = state_ind
        self.size = self.indices.size
        mat_slice = np.ix_(residual_ind, state_ind)
        self.Bx = fom.Bx[mat_slice]
        self.By = fom.By[mat_slice]
        self.C  = fom.C[mat_slice]
        self.I  = fom.Ixy[mat_slice]
        
        self.XY = fom.XY[self.indices]
        
    def set_initial(self, u0, v0):
        '''
        Set initial condition.
        
        inputs:
        u0:  function for initial u
        v0:  function for initial v
        '''
        self.u0 = u0(self.XY)
        self.v0 = v0(self.XY)
        
class subdomain_fom:
    '''
    Generate subdomain class for DD formulation of 2D Burgers' equation. 
    '''
    def __init__(self, fom, residual_ind, interior_ind, interface_ind, constraint_mat, in_ports, scaling=1):
        '''
        Initialize subdomain_fom class. 
        
        inputs: 
        fom:            instance of Burgers2D class. 
        residual_ind:   subdomain residual indices
        interior_ind:   subdomain interior indices
        interface_ind:  subdomain interface indices
        constraint_mat: constraint matrix for coupling conditions
        in_ports:       ports containing current subdomain
        
        methods:
        set_initial:    set initial data for initial value problem
        res_jac:        compute residual and residual jacobian on subdomain
        
        '''
        self.scaling = scaling
        self.constraint_mat = constraint_mat
        self.in_ports = in_ports
        self.residual_ind = residual_ind
        self.interior  = state_component(fom, self.residual_ind, interior_ind)
        self.interface = state_component(fom, self.residual_ind, interface_ind)
        self.spzero    = sp.csr_matrix((self.constraint_mat.shape[0], 2*self.interior.size))
    
    def set_initial(self, u0, v0):
        '''
        Set initial condition.
        
        inputs:
        u0:  function for initial u
        v0:  function for initial v
        '''
        self.interior.set_initial(u0, v0)
        self.interface.set_initial(u0, v0)
        
    def res_jac(self, un_intr, vn_intr, un_intf, vn_intf, 
                      uc_intr, vc_intr, uc_intf, vc_intf, lam, ht):
        '''
        Compute residual and residual jacobian.
        
        inputs: 
        un_intr:  interior u at next time step
        vn_intr:  interior v at next time step
        un_intf:  interface u at next time step
        vn_intf:  interface v at next time step
        uc_intr:  interior u at current time step
        vc_intr:  interior v at current time step
        uc_intf:  interface u at current time step
        vc_intf:  interface v at current time step
        lam:      lagrange multipliers
        ht:       time step
        
        ouputs:
        res: residual vector
        jac: jacobian matrix
        H:   Hessian submatrix for SQP solver
        rhs: RHS block vector in SQP solver
        Ax : constraint matrix times interface state
        '''
        start = time()
        
        # store relevant quantities
        un_res = self.interior.I@un_intr + self.interface.I@un_intf
        uc_res = self.interior.I@uc_intr + self.interface.I@uc_intf
        vn_res = self.interior.I@vn_intr + self.interface.I@vn_intf
        vc_res = self.interior.I@vc_intr + self.interface.I@vc_intf
        
        Bxu = self.interior.Bx@un_intr + self.interface.Bx@un_intf
        Byu = self.interior.By@un_intr + self.interface.By@un_intf
        Cu  = self.interior.C@un_intr + self.interface.C@un_intf
        
        Bxv = self.interior.Bx@vn_intr + self.interface.Bx@vn_intf
        Byv = self.interior.By@vn_intr + self.interface.By@vn_intf
        Cv  = self.interior.C@vn_intr + self.interface.C@vn_intf
        
        # compute residuals
        res_u = un_res - uc_res - ht*(un_res*Bxu + vn_res*Byu + Cu)
        res_v = vn_res - vc_res - ht*(un_res*Bxv + vn_res*Byv + Cv)
        res   = np.concatenate([res_u, res_v])
        
        # compute interior state jacobian
        UBx_VBy_C_intr = sp_diag(un_res)@self.interior.Bx + sp_diag(vn_res)@self.interior.By + self.interior.C
        Juu_intr = self.interior.I - ht*(sp_diag(Bxu)@self.interior.I + UBx_VBy_C_intr)
        Juv_intr = -ht*sp_diag(Byu)@self.interior.I
        Jvu_intr = -ht*sp_diag(Bxv)@self.interior.I
        Jvv_intr = self.interior.I - ht*(sp_diag(Byv)@self.interior.I + UBx_VBy_C_intr)
        jac_intr = sp.bmat([[Juu_intr, Juv_intr], [Jvu_intr, Jvv_intr]], format='csr')
        
        # compute interface state jacobian
        UBx_VBy_C_intf = sp_diag(un_res)@self.interface.Bx + sp_diag(vn_res)@self.interface.By + self.interface.C
        Juu_intf = self.interface.I - ht*(sp_diag(Bxu)@self.interface.I + UBx_VBy_C_intf)
        Juv_intf = -ht*sp_diag(Byu)@self.interface.I
        Jvu_intf = -ht*sp_diag(Bxv)@self.interface.I
        Jvv_intf = self.interface.I - ht*(sp_diag(Byv)@self.interface.I + UBx_VBy_C_intf)
        jac_intf = sp.bmat([[Juu_intf, Juv_intf], [Jvu_intf, Jvv_intf]], format='csr')
        
        # compute terms needed for SQP solver
        Ax  = self.constraint_mat@np.concatenate([un_intf, vn_intf])
        jac = sp.hstack([jac_intr, jac_intf])
        H   = self.scaling*(jac.T@jac)
        rhs = np.concatenate([self.scaling*(jac_intr.T@res),
                              self.scaling*(jac_intf.T@res) + self.constraint_mat.T@lam])
        
        return res, jac, H, rhs, Ax
    
class DD_FOM:
    '''
    Class for DD formulation of 2D Burgers equation.
    
    methods:
    set_initial:    set initial condition for all subdomains
    assemble_kkt:   function to assemble KKT system for Lagrange-Gauss-Newton solver
    solve:          solve 2D Burgers equation 
    '''
    def __init__(self, fom, nsub_x, nsub_y, constraint_type='strong', n_constraints=1, seed=None, scaling=1):
        self.x_lim = fom.x_lim
        self.y_lim = fom.y_lim
        self.nx  = fom.nx
        self.ny  = fom.ny
        self.nxy = fom.nxy
        self.hx  = fom.hx
        self.hy  = fom.hy
        self.hxy = fom.hxy
        self.n_sub = nsub_x*nsub_y
        
        self.scaling = self.hxy if scaling < 0 else scaling
        
        # generate subdomain indices for given number of subdomains
        indices = subdomain_indices(fom, nsub_x, nsub_y)
        
        # sort each interface node into a port
        port_array = np.zeros((len(indices.skeleton), self.n_sub), dtype=bool)
        for i, node in enumerate(indices.skeleton):
            for j in range(self.n_sub):
                port_array[i, j] = node in indices.interface[j]
        
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
            self.subdomain.append(subdomain_fom(fom, 
                                                indices.residual[i], 
                                                indices.interior[i],
                                                indices.interface[i], 
                                                constraint_mat[i], 
                                                in_ports[i], 
                                                scaling=self.scaling))
        A_list = []
        for s in self.subdomain: A_list += [s.spzero, s.constraint_mat] 
        self.A_block = sp.hstack(A_list, format='csr')
        
    def set_initial(self, u0, v0):
        '''
        Set initial condition.
        
        inputs:
        u0:  function for initial u
        v0:  function for initial v
        '''
        for s in self.subdomain:
            s.set_initial(u0, v0)
    
    def assemble_kkt(self, wn, wc, ht):
        '''
        Computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver. 
        
        inputs: 
        wn: vector of all interior and interface states for each subdomain 
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
            at the next time step
        wc: vector of interior and interface states for each subdomain and lagrange multipliers at current time step
        ht: time step
        
        outputs:
        rhs: RHS of the KKT system
        mat: KKT matrix
        runtime: "parallel" runtime to assemble KKT system
        '''
        start    = time()
        shift    = 0
        rhs      = []
        H_list   = []

        constraint_res = np.zeros(self.n_constraints)
        lam     = wn[-self.n_constraints:]
        runtime = time()-start
        stimes  = np.zeros(self.n_sub)
        
        for i, s in enumerate(self.subdomain):
            start = time()
            interior_ind  = np.arange(s.interior.indices.size)
            interface_ind = np.arange(s.interface.indices.size)

            un_intr = wn[interior_ind+shift]
            uc_intr = wc[interior_ind+shift]
            shift  += s.interior.indices.size
            vn_intr = wn[interior_ind+shift]
            vc_intr = wc[interior_ind+shift]
            shift  += s.interior.indices.size

            un_intf = wn[interface_ind+shift]
            uc_intf = wc[interface_ind+shift]
            shift  += s.interface.indices.size
            vn_intf = wn[interface_ind+shift]
            vc_intf = wc[interface_ind+shift]
            shift  += s.interface.indices.size
            
            # computes residual, jacobian, and other quantities needed for KKT system
            res, jac, H, rhs_i, Ax = s.res_jac(un_intr, vn_intr, un_intf, vn_intf, 
                                               uc_intr, vc_intr, uc_intf, vc_intf, lam, ht)
            stimes[i] = time()-start
            
            # RHS block for KKT system
            start = time()
            rhs.append(rhs_i)
            constraint_res += Ax
            
            H_list.append(H)
            runtime += time()-start
        
        start = time()
        rhs.append(constraint_res)
        rhs = np.concatenate(rhs)
        H_block = sp.block_diag(H_list)
        mat = sp.bmat([[H_block, self.A_block.T], [self.A_block, None]], format='csr')
        runtime += time()-start + stimes.max()
        
        return rhs, mat, runtime
    
    def solve(self, t_lim, nt, tol=1e-9, maxit=20, print_hist=False):
        '''
        Solve 2D Burgers' IVP. 
        
        inputs:
        t_lim:      t_lim[0] = initial time
                    t_lim[1] = final time 
        nt:         number of time steps
        tol:        [optional] solver relative tolerance. Default is 1e-10
        maxit:      [optional] maximum number of iterations. Default is 20
        print_hist: [optional] Set to True to print iteration history. Default is False
        
        outputs:
        uu:         list. uu[i] = array for u component of Burgers' equation solution on ith subdomain
        vv:         list. vv[i] = array for v component of Burgers' equation solution on ith subdomain
        runtime:    wall clock time for solve
        '''
        start = time()
        ht   = (t_lim[1]-t_lim[0])/nt
        ww    = np.zeros((nt+1, 
                          2*sum([s.interior.indices.size+s.interface.indices.size for s in self.subdomain]) +
                               self.n_constraints))
        wc    = []
        for s in self.subdomain: wc += [s.interior.u0, s.interior.v0, s.interface.u0, s.interface.v0]
        wc.append(np.zeros(self.n_constraints))
        wc    = np.concatenate(wc)
        ww[0] = wc

        runtime   = time()-start
        
        for k in range(nt):
            if print_hist: print(f'Time step {k}:')
            y, rh, norm_res, sh, iter, rtk, flag = newton_solve(lambda wn: self.assemble_kkt(wn, wc, ht),
                                                          wc, 
                                                          tol=tol, 
                                                          maxit=maxit, 
                                                          print_hist=print_hist)
            runtime += rtk
            
            start=time()
            wc = y
            ww[k+1] = wc
            if flag == 1: 
                print(f'Time step {k+1}: solver failed to converge in {maxit} iterations.')
                sys.stdout.flush()
                break
            elif flag == 2:
                print(f'Time step {k+1}: no stepsize found.')
                sys.stdout.flush()
                break
            runtime += time()-start
        
        # extract subdomain states from vector solution
        lam = ww[:, -self.n_constraints:]
        u_intr, v_intr, u_intf, v_intf = [], [], [], []
        u_full, v_full = np.zeros((nt+1, self.nxy)), np.zeros((nt+1, self.nxy))
        shift = 0
        for s in self.subdomain:
            u_intr_i = ww[:, shift:s.interior.indices.size+shift]
            u_full[:, s.interior.indices] = u_intr_i
            u_intr.append(u_intr_i)
            shift += s.interior.indices.size
            
            v_intr_i = ww[:, shift:s.interior.indices.size+shift]
            v_full[:, s.interior.indices] = v_intr_i
            v_intr.append(v_intr_i)
            shift += s.interior.indices.size
            
            u_intf_i = ww[:, shift:s.interface.indices.size+shift]
            u_full[:, s.interface.indices] = u_intf_i
            u_intf.append(u_intf_i)
            shift += s.interface.indices.size
            
            v_intf_i = ww[:, shift:s.interface.indices.size+shift]
            v_full[:, s.interface.indices] = v_intf_i
            v_intf.append(v_intf_i)
            shift += s.interface.indices.size
            
        return u_full, v_full, u_intr, v_intr, u_intf, v_intf, lam, runtime, flag
    
class soldict2class:
    '''
    Class to convert solution dictionary to data container.
    
    fields: 
    w:        rom solution 
    u:        u component solution
    v:        v component solution
    basis:    POD basis
    runtime:  runtime for computing solution
    '''
    def __init__(self, sol_dict):
        '''
        Instantiate soldict2class.
        
        inputs:
        sol_dict: dictionary with fields
                   runtime  = time required to compute solution
                   basis    = POD basis for reconstructing subdomain solution
                   solution = subdomain solution
        '''
        self.runtime = sol_dict['runtime']
        
        if 'basis' in sol_dict.keys():
            self.w     = sol_dict['solution'].T
            self.basis = sol_dict['basis'] 
            self.size  = self.basis.shape[0]//2
            uv = self.basis@self.w

        else: 
            uv = sol_dict['solution'].T
            self.size = uv.shape[0]//2
            
        self.u = uv[:self.size].T
        self.v = uv[self.size:].T
        
class subdomain_data:
    '''
    Class to store interior and interface solution data.
    
    fields:
    interior:    soldict2class instance corresponding to interior state solution
    interface:   soldict2class instance corresponding to interior state solution
    '''
    def __init__(self, intr_dict, intf_dict):
        '''
        Instantiate subdomain_data class
        
        inputs:
        intr_dict:    dictionary for interior state subdomain solution
        intf_dict:    dictionary for interface state subdomain solution
        '''
        self.interior  = soldict2class(intr_dict)
        self.interface = soldict2class(intf_dict)
        
class DDFOM_data:
    '''
    Data container class for FOM solution. Loads DD FOM data corresponding to given discretization.
    
    fields:  
    nx:        number of spatial grid points in x direction
    ny:        number of spatial grid points in y direction
    nxy:       total discretization size
    nt:        number of time steps
    viscosity: viscosity
    mu:        parameter for initial condition
    nsub_x:    number of subdomains in x direction
    nsub_y:    number of subdomains in y direction
    nsub:      total number of subdomains
    subdomain: list of subdomains with interior and interface solution data
    
    methods:   
    assemble_full_solution:  assemble solution on full domain with subdomain data
    '''
    def __init__(self, nx, ny, nt, viscosity, mu, nsub_x, nsub_y, data_dir):
        '''
        Instantiate fom_data class.
        
        inputs:
        nx:        number of spatial grid points in x direction
        ny:        number of spatial grid points in y direction
        nxy:       total discretization size
        nt:        number of time steps
        viscosity: viscosity
        mu:        parameter for initial condition
        nsub_x:    number of subdomains in x direction
        nsub_y:    number of subdomains in y direction
        data_dir:  directory containing solution data
        '''
        self.nx   = nx
        self.ny   = ny
        self.nxy  = nx*ny
        self.nt   = nt
        self.viscosity = viscosity
        self.mu   = mu
        self.nsub_x = nsub_x
        self.nsub_y = nsub_y
        self.nsub = nsub_x*nsub_y
        
        # load DD FOM data
        data_dir2 = data_dir + f'nx_{nx}_ny_{ny}_nt_{nt}_visc_{viscosity}/DD_{nsub_x}x_by_{nsub_y}y/' 
        self.subdomain = []
        for i in range(self.nsub):
            sub_dir = data_dir2 + f'sub_{i+1}of{self.nsub}/'
            intr_dict = pickle.load(open(sub_dir + f'interior/mu_{mu}_uv_state.p', 'rb'))
            intf_dict = pickle.load(open(sub_dir + f'interface/mu_{mu}_uv_state.p', 'rb'))
            
            self.subdomain.append(subdomain_data(intr_dict, intf_dict))
            
        self.runtime = self.subdomain[0].interior.runtime
        self.nt = nt
    def assemble_full_solution(self, ddfom):
        '''
        Assemble full domain solution from subdomain data.
        
        inputs: 
        ddfom:     DD FOM class 
        '''
        self.u_full = np.zeros((self.nt+1, self.nxy))
        self.v_full = np.zeros((self.nt+1, self.nxy))
        
        for i,s in enumerate(ddfom.subdomain):
            self.u_full[:, s.interior.indices] = self.subdomain[i].interior.u
            self.v_full[:, s.interior.indices] = self.subdomain[i].interior.v
            self.u_full[:, s.interface.indices] = self.subdomain[i].interface.u
            self.v_full[:, s.interface.indices] = self.subdomain[i].interface.v
        
        self.UU = self.u_full.reshape(self.nt+1, self.nx, self.ny)
        self.VV = self.v_full.reshape(self.nt+1, self.nx, self.ny)        