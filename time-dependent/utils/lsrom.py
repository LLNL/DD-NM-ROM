import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from time import time
from utils.solvers import newton_solve
from utils.helpers import sp_diag, select_sample_nodes
import dill as pickle
import sys

class lsrom_component:
    '''
    Generate LS-ROM state component for 2D Burgers' equation from basis
    
    input:
    component:    instance of FOM state_component class
    basis:        POD basis corresponding to state component
    hr_nodes:     [optional] hyper reduction nodes for current state component
    cross_nodes:  [optional] hyper reduction nodes for other state component
    
    fields:
    basis:        POD basis
    size:         size of POD basis
    Bx:           ROM Bx matrix
    By:           ROM By matrix
    C:            ROM C matrix
    I:            ROM inclusion matrix corresponding to current residual component
    Io:           ROM inclusion matrix corresponding to other residual component
    
    '''
    def __init__(self, component, basis, hr_nodes=[], cross_nodes=[]):
        self.basis   = basis
        self.size    = self.basis.shape[1]
        
        if len(hr_nodes)>0:
            self.hr_nodes = hr_nodes
            self.Bx       = component.Bx[hr_nodes]@basis
            self.By       = component.By[hr_nodes]@basis
            self.C        = component.C[hr_nodes]@basis
            self.I        = component.I[hr_nodes]@basis
            self.Io       = component.I[cross_nodes]@basis
        else:
            self.Bx      = component.Bx@basis
            self.By      = component.By@basis
            self.C       = component.C@basis
            self.I       = component.I@basis
            
class burgers_rom_component:
    '''
    Generate ROM state component for 2D Burgers' equation from basis
    
    fields: 
    indices:      indices of FOM state corresponding to component
    basis:        POD basis
    basis_size:   size of POD basis
    XY:           FD grid points corresponding to current state component
    u:            lsrom_component instance corresponding to u states
    v:            lsrom_component instance corresponding to v states
    
    methods:
    set_initial: set initial condition for current state
    '''
    def __init__(self, component, basis, res_size=-1, hr_nodes=[]):
        '''
        Initialize burgers_rom_component class.
        
        inputs:
        component:    instance of FOM state_component class
        basis:        POD basis corresponding to state component
        hr_nodes:     [optional] residual nodes selected for HR. If HR is not used, hr_nodes = []
        '''
        self.basis      = basis
        self.basis_size = basis.shape[1]
        self.XY         = component.XY
        self.indices    = component.indices
        if len(hr_nodes) > 0:
            hr_nodes_u = hr_nodes[hr_nodes < res_size]
            hr_nodes_v = hr_nodes[hr_nodes >= res_size]-res_size
        else:
            hr_nodes_u, hr_nodes_v = [], []
            
        self.u = lsrom_component(component, basis[:component.size], hr_nodes=hr_nodes_u, cross_nodes=hr_nodes_v)
        self.v = lsrom_component(component, basis[component.size:], hr_nodes=hr_nodes_v, cross_nodes=hr_nodes_u)
    
    def set_initial(self, u0, v0):
        '''
        Set initial condition.
        
        inputs:
        u0:  function for initial u
        v0:  function for initial v
        '''
        self.w0 = self.basis.T@np.concatenate([u0(self.XY), v0(self.XY)])
    
        
def assemble_snapshot_matrix(directory, parameters, name_gen=None):
    '''
    Assemble snapshot matrix of 2D Burgers' equation solutions for use in POD basis computation.
    
    inputs:
    directory:    directory containing data
    parameters:   parameters of desired snapshot data
    handle_gen:   function for generating filenames given parameter values
    
    output:
    data:         matrix containing snapshot data
    '''
    if name_gen == None: name_gen = lambda p: f'mu_{p}_uv_state.p'
    
    parameters = [parameters] if type(parameters) != list else parameters
    data = []
    for p in parameters:
        uv = pickle.load(open(directory + name_gen(p), 'rb'))
        data.append(uv['solution'])
    
    return np.vstack(data).T

def save_svd_data(data, filename):
    '''
    Save SVD data required to compute POD basis. 
    Saves dictionary with fields:
        left_vecs:   matrix of left singular vectors
        sing_vals:   vector of singular values
        
    inputs: 
    data:      data matrix for computing svd
    filename:  filename for saved data
    '''
    U, S, Yh  = la.svd(data, full_matrices=False, check_finite=False)
    svd_dict = {'left_vecs': U, 'sing_vals': S}
    pickle.dump(svd_dict, open(filename, 'wb'))
    
def save_full_subdomain_svd(intr_data, intf_data, intr_filename, intf_filename):
    '''
    Save SVD data for full subdomain snapshots required to compute POD basis. 
    Saves dictionaries with fields:
        left_vecs:   matrix of left singular vectors
        sing_vals:   vector of singular values
        
    inputs: 
    intr_data:      interior state data matrix for computing svd
    intr_filename:  filename for saved interior state data
    intf_data:      interface state data matrix for computing svd
    intf_filename:  filename for saved interface state data
    '''
    intr_size = intr_data.shape[0]
    full_data = np.vstack([intr_data, intf_data])
    
    U, S, Yh  = la.svd(full_data, full_matrices=False, check_finite=False)
    
    intr_dict = {'left_vecs': U[:intr_size], 'sing_vals': S}
    pickle.dump(intr_dict, open(intr_filename, 'wb'))
    
    intf_dict = {'left_vecs': U[intr_size:], 'sing_vals': S}
    pickle.dump(intf_dict, open(intf_filename, 'wb'))
    
    
# compute POD bases given SVD data
def compute_bases_from_svd(data_dict,
                           ec=1e-8, 
                           nbasis=-1):
    '''
    Computes POD bases given saved SVD data.
    
    inputs:
    data_dict: dictionary with fields
                'left_vecs' : left singular vectors of snapshot data for each subdomain
                'sing_vals' : singular values of snapshot data for each subdomain
    ec: [optional] energy criterior for choosing size of basis. Default is 1e-8
    nbasis: [optional] size of basis. Setting to -1 uses energy criterion. Default is -1.
    
    output:
    basis: POD basis
    
    '''
    U = data_dict['left_vecs']
    if nbasis <= 0:
        s = data_dict['sing_vals']
        ss = s*s
        dim = np.where(np.array([np.sum(ss[0:j+1]) for j in range(s.size)])/np.sum(ss) >= 1-ec)[0].min()+1
    else:
        dim = nbasis
    return U[:, :dim]
    
# compute POD basis for given snapshot data
def POD(data, ec=1e-5, n_basis=-1):
    '''
    Compute POD basis given snapshot data.
    inputs:
        data: (N, M)-array of snapshot data
        ec: [optional] energy criterion for selecting number of basis vectors. Default is 1e-5
        n_basis: [optional] specify number of basis vectors. The default of -1 uses the energy criterion. 
                 Setting n_basis > 0 overrides the energy criterion.

    outputs:
        U: (N, n_basis)-array containing POD basis
        s: singular values of data
    '''
    u, s, vh = la.svd(data, full_matrices=False, check_finite=False)
    if n_basis <= 0:
        ss = s*s
        n_basis = np.where(np.array([np.sum(ss[0:j+1]) for j in range(s.size)])/np.sum(ss) >= 1-ec)[0].min()+1
    return u[:, 0:n_basis], s

def assemble_port_bases(port_bases, ddfom):
    assert len(port_bases) == len(ddfom.ports)
            
    # assign coupling conditions to ROM states
    # rom_port_ind[j][p] = indices of intf state on subdomain j corresponding to port p
    rom_port_ind = []
    for s in ddfom.subdomain:
        rom_port_dict = {}
        shift = 0
        for p in s.in_ports:
            rom_port_dict[p] = np.arange(port_bases[p].shape[1])+shift
            shift += port_bases[p].shape[1]
        rom_port_ind.append(rom_port_dict)   

    # assemble interface bases using port bases
    interface_bases = []
    n_intf_list = []
    for i, s in enumerate(ddfom.subdomain):
        n_intf = np.sum([len(rom_port_ind[i][p]) for p in s.in_ports])
        n_intf_list.append(n_intf)
        basis  = np.zeros((2*s.interface.size, n_intf))
        for p in s.in_ports:
            p_ind = np.concatenate([ddfom.port_dict[ddfom.ports[p]], ddfom.port_dict[ddfom.ports[p]]+ddfom.nxy])
            s_ind = np.concatenate([s.interface.indices, s.interface.indices+ddfom.nxy])
            fomport2intf = np.nonzero(np.isin(s_ind, p_ind))[0]
            basis[np.ix_(fomport2intf, rom_port_ind[i][p])] += port_bases[p]
        interface_bases.append(basis)

    # assemble ROM-port constraint matrices
    n_constraints  = np.sum([(len(port)-1)*port_bases[k].shape[1] for k, port in enumerate(ddfom.ports)])
    constraint_mat_list = [sp.coo_matrix((n_constraints, n_intf)) for n_intf in n_intf_list]

    shift = 0 
    for j, p in enumerate(ddfom.ports):
        port = list(p)
        npj = port_bases[j].shape[1]
        for i in range(len(port)-1):
            s1   = port[i]
            constraint_mat_list[s1].col  = np.concatenate((constraint_mat_list[s1].col, rom_port_ind[s1][j]))
            constraint_mat_list[s1].row  = np.concatenate((constraint_mat_list[s1].row, np.arange(npj)+shift))
            constraint_mat_list[s1].data = np.concatenate((constraint_mat_list[s1].data, np.ones(npj)))   

            s2   = port[i+1]
            constraint_mat_list[s2].col  = np.concatenate((constraint_mat_list[s2].col, rom_port_ind[s2][j]))
            constraint_mat_list[s2].row  = np.concatenate((constraint_mat_list[s2].row, np.arange(npj)+shift))
            constraint_mat_list[s2].data = np.concatenate((constraint_mat_list[s2].data, -np.ones(npj)))

            shift += npj
    return interface_bases, constraint_mat_list, n_constraints

class subdomain_rom:
    '''
    Generate ROM subdomain class for DD formulation of 2D Burgers' equation.
    
    fields:
    constraint_mat:    compatibility constraint matrix
    in_ports:          list of ports that current subdomain belongs to
    residual_ind:      indices of FOM corresponding to residual states in current subdomain
    interior:          instance of burgers_rom_component class corresponding to interior states
    interface:         instance of burgers_rom_component class corresponding to interface states
    spzero:            sparse zero matrix for constructing KKT system
    
    methods:
    set_initial:       set initial condition for current subdomain
    res_jac:           compute residual and residual jacobian
    '''
    def __init__(self, subdomain, intr_basis, intf_basis, constraint_mat, scaling=1):
        '''
        initialize subdomain rom class.
        
        inputs:
        subdomain:       instance of subdomain_fom class corresponding to current subdomain
        intr_basis:      basis for interior states
        intf_basis:      basis for interface states
        constraint_mat:  constraint matrix corresponding to current subdomain
        '''
        self.in_ports       = subdomain.in_ports
        self.residual_ind   = subdomain.residual_ind
        self.constraint_mat = constraint_mat
        self.scaling        = scaling
        
        self.interior  = burgers_rom_component(subdomain.interior, intr_basis)
        self.interface = burgers_rom_component(subdomain.interface, intf_basis)
        
        self.spzero = sp.csr_matrix((self.constraint_mat.shape[0], self.interior.basis_size))
    
    def set_initial(self, u0, v0):
        '''
        Set initial condition.
        
        inputs:
        u0:  function for initial u
        v0:  function for initial v
        '''
        self.interior.set_initial(u0, v0)
        self.interface.set_initial(u0, v0)
        
    def res_jac(self, wn_intr, wn_intf, wc_intr, wc_intf, lam, ht):
        '''
        Compute residual and residual jacobian.
        
        inputs: 
        wn_intr:  interior w at next time step
        wn_intf:  interface w at next time step
        wc_intr:  interior w at current time step
        wc_intf:  interface w at current time step
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
        un_res = self.interior.u.I@wn_intr + self.interface.u.I@wn_intf
        uc_res = self.interior.u.I@wc_intr + self.interface.u.I@wc_intf
        vn_res = self.interior.v.I@wn_intr + self.interface.v.I@wn_intf
        vc_res = self.interior.v.I@wc_intr + self.interface.v.I@wc_intf
        
        Bxu = self.interior.u.Bx@wn_intr + self.interface.u.Bx@wn_intf
        Byu = self.interior.u.By@wn_intr + self.interface.u.By@wn_intf
        Cu  = self.interior.u.C@wn_intr + self.interface.u.C@wn_intf
        
        Bxv = self.interior.v.Bx@wn_intr + self.interface.v.Bx@wn_intf
        Byv = self.interior.v.By@wn_intr + self.interface.v.By@wn_intf
        Cv  = self.interior.v.C@wn_intr + self.interface.v.C@wn_intf
        
        # compute residuals
        res_u = un_res - uc_res - ht*(un_res*Bxu + vn_res*Byu + Cu)
        res_v = vn_res - vc_res - ht*(un_res*Bxv + vn_res*Byv + Cv)
        res   = np.concatenate([res_u, res_v])
        
        # compute interior state jacobian
        Ju_intr = self.interior.u.I \
                  - ht*(sp_diag(Bxu)@self.interior.u.I + sp_diag(un_res)@self.interior.u.Bx
                            + sp_diag(Byu)@self.interior.v.I + sp_diag(vn_res)@self.interior.u.By
                            + self.interior.u.C)
        Jv_intr = self.interior.v.I \
                  - ht*(sp_diag(Bxv)@self.interior.u.I + sp_diag(un_res)@self.interior.v.Bx
                            + sp_diag(Byv)@self.interior.v.I + sp_diag(vn_res)@self.interior.v.By
                            + self.interior.v.C)
        
        # compute interface state jacobian
        Ju_intf = self.interface.u.I \
                  - ht*(sp_diag(Bxu)@self.interface.u.I + sp_diag(un_res)@self.interface.u.Bx
                            + sp_diag(Byu)@self.interface.v.I + sp_diag(vn_res)@self.interface.u.By
                            + self.interface.u.C)
        Jv_intf = self.interface.v.I \
                  - ht*(sp_diag(Bxv)@self.interface.u.I + sp_diag(un_res)@self.interface.v.Bx
                            + sp_diag(Byv)@self.interface.v.I + sp_diag(vn_res)@self.interface.v.By
                            + self.interface.v.C)
        
        jac_intr = np.vstack([Ju_intr, Jv_intr])
        jac_intf = np.vstack([Ju_intf, Jv_intf])
        
        # compute terms needed for SQP solver
        Ax  = self.constraint_mat@wn_intf
        jac = np.hstack([jac_intr, jac_intf])
        H   = self.scaling*(jac.T@jac)
        rhs = np.concatenate([self.scaling*(jac_intr.T@res),
                              self.scaling*(jac_intf.T@res) + self.constraint_mat.T@lam])
        
        return res, jac, H, rhs, Ax

class subdomain_rom_hr(subdomain_rom):
    '''
    Generate hyper reduced ROM subdomain class for DD formulation of 2D Burgers' equation.
    
    fields:
    ht:                time step
    fom_constraint:    FOM constaint matrix multiplied by interface basis
    constraint_mat:    weak compatibility constraint matrix
    in_ports:          list of ports that current subdomain belongs to
    residual_ind:      indices of FOM corresponding to residual states in current subdomain
    hr_nodes:          indices corresponding to HR nodes
    interior:          instance of burgers_rom_component class corresponding to interior states
    interface:         instance of burgers_rom_component class corresponding to interface states
    spzero:            sparse zero matrix for constructing KKT system
    
    methods:
    set_initial:       set initial condition for current subdomain
    res_jac:           compute residual and residual jacobian
    '''
    
    def __init__(self, 
                 subdomain, 
                 intr_basis, 
                 intf_basis, 
                 res_basis,
                 constraint_mat,
                 nz, ncol, n_corners=5, scaling=1):
        '''
        Initialize hyper-reduced subdomain rom class.
        
        inputs:
        subdomain:       instance of subdomain_fom class corresponding to current subdomain
        intr_basis:      basis for interior states
        intf_basis:      basis for interface states
        res_basis:       basis for residual to be used in computation of HR nodes
        constraint_mat:  constraint matrix corresponding to present subdomain
        nz:              number of hyper reduction indices
        ncol:            number of working columns for sample node selection algorithm
        n_corners:       [optional] number of interface columns to include in sample nodes. Default is 5
        '''
        self.in_ports       = subdomain.in_ports
        self.residual_ind   = subdomain.residual_ind
        self.constraint_mat = constraint_mat
        self.scaling        = scaling
        
        res_size = res_basis.shape[0]//2
        self.hr_nodes  = select_sample_nodes(subdomain, res_basis, res_size, nz, ncol, n_corners=n_corners)

        self.interior  = burgers_rom_component(subdomain.interior, intr_basis, res_size=res_size, hr_nodes=self.hr_nodes)
        self.interface = burgers_rom_component(subdomain.interface, intf_basis, res_size=res_size, hr_nodes=self.hr_nodes)
        
        self.spzero = sp.csr_matrix((self.constraint_mat.shape[0], self.interior.basis_size))
        
    def res_jac(self, wn_intr, wn_intf, wc_intr, wc_intf, lam, ht):
        '''
        Compute residual and residual jacobian.
        
        inputs: 
        wn_intr:  interior w at next time step
        wn_intf:  interface w at next time step
        wc_intr:  interior w at current time step
        wc_intf:  interface w at current time step
        lam:      lagrange multipliers
        ht:       time step
        
        ouputs:
        res: residual vector
        jac: jacobian matrix
        H:   Hessian submatrix for SQP solver
        rhs: RHS block vector in SQP solver
        Ax : constraint matrix times interface state
        '''
        
        # store relevant quantities
        uun_res = self.interior.u.I@wn_intr  + self.interface.u.I@wn_intf
        uvn_res = self.interior.v.Io@wn_intr + self.interface.v.Io@wn_intf
        vun_res = self.interior.u.Io@wn_intr + self.interface.u.Io@wn_intf
        vvn_res = self.interior.v.I@wn_intr  + self.interface.v.I@wn_intf
        
        uuc_res = self.interior.u.I@wc_intr + self.interface.u.I@wc_intf
        vvc_res = self.interior.v.I@wc_intr + self.interface.v.I@wc_intf
        
        Bxu = self.interior.u.Bx@wn_intr + self.interface.u.Bx@wn_intf
        Byu = self.interior.u.By@wn_intr + self.interface.u.By@wn_intf
        Cu  = self.interior.u.C@wn_intr + self.interface.u.C@wn_intf
        
        Bxv = self.interior.v.Bx@wn_intr + self.interface.v.Bx@wn_intf
        Byv = self.interior.v.By@wn_intr + self.interface.v.By@wn_intf
        Cv  = self.interior.v.C@wn_intr + self.interface.v.C@wn_intf
        
        # compute residuals
        res_u = uun_res - uuc_res - ht*(uun_res*Bxu + uvn_res*Byu + Cu)
        res_v = vvn_res - vvc_res - ht*(vun_res*Bxv + vvn_res*Byv + Cv)
        res   = np.concatenate([res_u, res_v])
        
        # compute interior state jacobian
        Ju_intr = self.interior.u.I \
                  - ht*(sp_diag(Bxu)@self.interior.u.I + sp_diag(uun_res)@self.interior.u.Bx
                            + sp_diag(Byu)@self.interior.v.Io + sp_diag(uvn_res)@self.interior.u.By
                            + self.interior.u.C)
        Jv_intr = self.interior.v.I \
                  - ht*(sp_diag(Bxv)@self.interior.u.Io + sp_diag(vun_res)@self.interior.v.Bx
                            + sp_diag(Byv)@self.interior.v.I + sp_diag(vvn_res)@self.interior.v.By
                            + self.interior.v.C)
        
        # compute interface state jacobian
        Ju_intf = self.interface.u.I \
                  - ht*(sp_diag(Bxu)@self.interface.u.I + sp_diag(uun_res)@self.interface.u.Bx
                            + sp_diag(Byu)@self.interface.v.Io + sp_diag(uvn_res)@self.interface.u.By
                            + self.interface.u.C)
        Jv_intf = self.interface.v.I \
                  - ht*(sp_diag(Bxv)@self.interface.u.Io + sp_diag(vun_res)@self.interface.v.Bx
                            + sp_diag(Byv)@self.interface.v.I + sp_diag(vvn_res)@self.interface.v.By
                            + self.interface.v.C)
        
        jac_intr = np.vstack([Ju_intr, Jv_intr])
        jac_intf = np.vstack([Ju_intf, Jv_intf])
        
        # compute terms needed for SQP solver
        Ax  = self.constraint_mat@wn_intf
        jac = np.hstack([jac_intr, jac_intf])
        H   = self.scaling*(jac.T@jac)
        rhs = np.concatenate([self.scaling*(jac_intr.T@res),
                              self.scaling*(jac_intf.T@res) + self.constraint_mat.T@lam])
        
        return res, jac, H, rhs, Ax
    
class DD_LSROM:
    '''
    Class for DD LS-ROM of 2D Burgers equation.
    
    fields:
    n_constraints:   number of (equality) constraints for NLP
    subdomain:       list of subdomain_LS_ROM or subdomain_LS_ROM_HR classes corresponding to each subdomain, i.e.
                     subdomain[i] = reduced subdomain class corresponding to subdomain [i]
    
               
    methods:
    set_initial:    set initial condition for all subdomains
    assemble_kkt:   function to assemble KKT system for Lagrange-Gauss-Newton solver
    solve:          solve 2D Burgers equation 
    '''
    def __init__(self, 
                 ddfom, 
                 intr_bases, 
                 intf_bases, 
                 res_bases=[],
                 port_bases=[],
                 constraint_type='srpc',
                 hr=False,
                 sample_ratio=2,
                 n_samples=-1,
                 n_corners=-1,
                 n_constraints=1, 
                 seed=None, scaling=1):
        '''
        Initialize DD_LSROM class.
        
        inputs:
        ddfom:         instance of DD_FOM class
        intr_bases:    list of interior state POD bases for each subdomain 
        intf_bases:    list of interface state POD bases for each subdomain 
        res_bases:     [optional] list of residual POD bases for each subdomain. Used for computed HR nodes. 
        hr:            [optional] boolean to indicate if HR is to be applied. Default is False
        sample_ratio:  [optional] ratio of number of hyper-reduction samples to residual basis size. Default is 2
        n_samples:     [optional] specify number of hyper reduction sample nodes. 
                          If n_samples is an array with length equal to the number of subdomains, then 
                          n_samples[i] is the number of HR samples on the ith subdomain.
                
                          If n_samples is a positive integer, then each subdomain has n_samples HR nodes. 
                
                          Otherwise, the number of samples is determined by the sample ratio. 
                          Default is -1. 
                
        n_corners:     [optional] Number of interface nodes included in the HR sample nodes. 
                          If n_corners is an array with length equal to the number of subdomains, then 
                          n_corners[i] is the number of interface HR nodes on the ith subdomain.
                 
                          If n_corners is a positive integer, then each subdomain has n_corners interface HR nodes. 
                
                          Otherwise, the number of interface HR nodes on each subdomain is determined by n_samples
                          multiplied by the ratio of the number of interface nodes contained in the residual nodes 
                          to the total number of residual nodes.
                
                          Default is -1. 
        '''
        
        self.nxy   = ddfom.nxy
        self.hxy   = ddfom.hxy
        self.n_sub = ddfom.n_sub
        self.ports = ddfom.ports
        self.scaling = self.hxy if scaling<0 else scaling
        
        # assign interface bases and constraints by constraint type
        # strong rom-port constraints
        if constraint_type=='srpc':
            intf_bases, constraint_mat_list, self.n_constraints = assemble_port_bases(port_bases, ddfom)
            for k, b in enumerate(intf_bases): print(f'interface_basis[{k}].shape={b.shape}')
                
        # weak fom-port constraints
        else: 
            self.n_constraints = n_constraints
            rng = np.random.default_rng(seed)
            self.constraint_mult = rng.standard_normal(size=(self.n_constraints, ddfom.n_constraints))
            constraint_mat_list  = [self.constraint_mult@s.constraint_mat@intf_bases[j] for j, s in enumerate(ddfom.subdomain)]
            
        # instantiate subdomain classes
        self.subdomain = []
        if hr:
            # compute parameters for hyper reduction
            ncol = np.array([rb.shape[1] for rb in res_bases])             # number of working columns per subdomain
            nz_max = np.array([rb.shape[0] for rb in res_bases])
            
            # number of sample nodes per subdomain
            if isinstance(n_samples, int):
                if n_samples > 0:
                    nz = n_samples*np.ones(self.n_sub)
                else:
                    nz = sample_ratio*ncol
            else:
                nz = n_samples
            nz = np.maximum(np.minimum(nz, nz_max), ncol)
            
            # number of corner nodes per subdomain
            n_corners_max = nz_max - np.array([b.shape[0] for b in intr_bases])
            if isinstance(n_corners, int):
                if n_corners > 0:
                    n_corners = n_corners*np.ones(self.n_sub)
                else:
                    n_corners = np.round(nz*n_corners_max/nz_max)
            n_corners = np.minimum(n_corners, n_corners_max)
            
            for i, s in enumerate(ddfom.subdomain):
                self.subdomain.append(subdomain_rom_hr(s, 
                                                       intr_bases[i], 
                                                       intf_bases[i], 
                                                       res_bases[i],
                                                       constraint_mat_list[i],
                                                       int(nz[i]),
                                                       int(ncol[i]),
                                                       n_corners=int(n_corners[i]),
                                                       scaling=self.scaling))
        else:
            for i, s in enumerate(ddfom.subdomain):
                self.subdomain.append(subdomain_rom(s, intr_bases[i], intf_bases[i], constraint_mat_list[i], scaling=self.scaling))
                    
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
        wn: vector of all reduced interior and interface states for each subdomain 
            and the lagrange multipliers lam in the order
                    w = [w_intr[0], 
                         w_intf[0],
                         ..., 
                         w_intr[n_sub], 
                         w_intf[n_sub],
                         lam]
            at the next time step
        wc: vector of reduced interior and interface states for each subdomain 
            and lagrange multipliers at current time step
            
        outputs:
        rhs: RHS of the KKT system
        mat: KKT matrix
        runtime: "parallel" runtime to assemble KKT system
        '''
        start    = time()
        shift    = 0
        rhs      = []
        H_list   = []
        A_list   = []
        
        constraint_res = np.zeros(self.n_constraints)
        lam     = wn[-self.n_constraints:]
        runtime = time()-start
        stimes  = np.zeros(self.n_sub)
        
        for i, s in enumerate(self.subdomain):
            start = time()
            interior_ind  = np.arange(s.interior.basis_size)
            interface_ind = np.arange(s.interface.basis_size)

            wn_intr = wn[interior_ind+shift]
            wc_intr = wc[interior_ind+shift]
            shift  += s.interior.basis_size

            wn_intf = wn[interface_ind+shift]
            wc_intf = wc[interface_ind+shift]
            shift  += s.interface.basis_size
            
            # computes residual, jacobian, and other quantities needed for KKT system
            res, jac, H, rhs_i, Ax = s.res_jac(wn_intr, wn_intf, 
                                               wc_intr, wc_intf, lam, ht)
            stimes[i] = time()-start
            
            # RHS block for KKT system
            start = time()
            rhs.append(rhs_i)
            constraint_res += Ax
            A_list += [s.spzero, s.constraint_mat] 
            H_list.append(H)
            runtime += time()-start
        
        start = time()
        rhs.append(constraint_res)
        rhs = np.concatenate(rhs)
        H_block = sp.block_diag(H_list)
        A_block = sp.hstack(A_list, format='csr')
        mat = sp.bmat([[H_block, A_block.T], [A_block, None]], format='csr')
        runtime += time()-start + stimes.max()
        
        return rhs, mat, runtime
    
    def solve(self, t_lim, nt, tol=1e-9, maxit=20, print_hist=False):
        '''
        Solve DD LS-ROM for 2D Burgers' IVP. 
        
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
        ht    = (t_lim[1]-t_lim[0])/nt
        wc    = []
        for s in self.subdomain: wc += [s.interior.w0, s.interface.w0]
        wc.append(np.zeros(self.n_constraints))
        wc    = np.concatenate(wc)
        ww = [wc]
        iter_hist = []

        runtime   = time()-start
        for k in range(nt):
            start = time()
            if print_hist: print(f'Time step {k}:')
            runtime += time()-start

            y, rh, norm_res, sh, iter, rtk, flag = newton_solve(lambda wn: self.assemble_kkt(wn, wc, ht),
                                                                wc, 
                                                                tol=tol, 
                                                                maxit=maxit, 
                                                                print_hist=print_hist)
            runtime += rtk
            iter_hist.append(iter)
            
            start=time()
            wc = y
            ww.append(y)
            if flag == 1: 
                print(f'Time step {k+1}: solver failed to converge in {maxit} iterations.')
                print(f'                 terminal gradient norm = {norm_res[-1]:1.4e}')
                sys.stdout.flush()
                break
            elif flag == 2:
                print(f'Time step {k+1}: no stepsize found.')
                sys.stdout.flush()
                break
            runtime += time()-start
        
        # extract subdomain states from vector solution
        ww  = np.vstack(ww)
        lam = ww[:, -self.n_constraints:]
        w_intr, w_intf = [], []
        u_intr, v_intr, u_intf, v_intf = [], [], [], []
        u_full, v_full = np.zeros((ww.shape[0], self.nxy)), np.zeros((ww.shape[0], self.nxy))
        shift = 0
        for s in self.subdomain:
            w_intr_i = ww[:, shift:s.interior.basis_size+shift].T
            u_intr_i = s.interior.u.basis@w_intr_i
            v_intr_i = s.interior.v.basis@w_intr_i
            u_full[:, s.interior.indices] = u_intr_i.T
            v_full[:, s.interior.indices] = v_intr_i.T
            w_intr.append(w_intr_i.T)
            u_intr.append(u_intr_i.T)
            v_intr.append(v_intr_i.T)
            shift += s.interior.basis_size
            
            w_intf_i = ww[:, shift:s.interface.basis_size+shift].T
            u_intf_i = s.interface.u.basis@w_intf_i
            v_intf_i = s.interface.v.basis@w_intf_i
            u_full[:, s.interface.indices] = u_intf_i.T
            v_full[:, s.interface.indices] = v_intf_i.T
            w_intf.append(w_intf_i.T)
            u_intf.append(u_intf_i.T)
            v_intf.append(v_intf_i.T)
            shift += s.interface.basis_size
            
        return w_intr, w_intf, u_intr, v_intr, u_intf, v_intf, u_full, v_full, lam, runtime, iter_hist, flag