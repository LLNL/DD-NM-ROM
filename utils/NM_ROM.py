# NM_ROM.py
# Author: Alejandro Diaz

import numpy as np
import scipy.sparse as sp
import torch
from copy import copy
from time import time 
from utils.LS_ROM import select_sample_nodes, POD
from utils.newton_solve import newton_solve_combinedFJac as newton_solve
import scipy.linalg as la
from scipy.interpolate import RBFInterpolator

def separate_snapshots(ddmdl, 
                       snapshots,
                       residuals=None):
    '''
    Separate full-domain snapshot data into interior and interface snapshots on each subdomain.
    
    inputs: 
    ddmdl: instance of DD_model class
    snapshots: list where snapshots[i] = snapshot vector corresponding to ith parameter set
    residuals: [optional] list where residuals[i][j] = Newton residual at jth iteration correpsonding to ith parameter set
                          if residuals == None, then the residual basis is not computed.
    
    outputs:
    interior_bases: list where interior_bases[i] is interior basis for ith subdomain
    interface_bases: list where interface_bases[i] is interface basis for ith subdomain
    residual_bases: list where residual_bases[i] is residual basis for ith subdomain
    
    '''
    
    Ns = len(snapshots)
    
    # separate full-domain snapshot data into subdomain snapshot data
    # The separated data corresponding to each subdomain and parameter pair is organized as follows:
    # - interior[i][j] = interior state snapshot on subdomain $i$ corresponding to parameters Mu[j]
    # - interface[i][j] = interface state snapshot on subdomain $i$ corresponding to parameters Mu[j]
    # - residual[i][j][k] = $k$th residual snapshot on subdomain $i$ corresponding to parameters Mu[j]
    
    # collect interior and interface snapshots on each subdomain
    interior = []
    interface = []
    for s in ddmdl.subdomain:
        interior_i = []
        interface_i = []
        residual_i = []
        for j in range(Ns):
            interior_i.append(np.concatenate([snapshots[j][s.interior_ind], 
                                              snapshots[j][s.interior_ind+ddmdl.nxy]]))
            interface_i.append(np.concatenate([snapshots[j][s.interface_ind], 
                                              snapshots[j][s.interface_ind+ddmdl.nxy]]))
        interior.append(interior_i)
        interface.append(interface_i)
        
    if residuals != None:
        # collect residual snapshots on each subdomain
        Nr = len(residuals)
        residual = []
        for s in ddmdl.subdomain:
            residual_i = []
            for j in range(Nr):
                residual_i.append(np.hstack([residuals[j][:, s.residual_ind], 
                                             residuals[j][:, s.residual_ind+ddmdl.nxy]]))
            residual.append(residual_i)

    if residuals != None:
        return interior, interface, residual
    else:
        return interior, interface
    
class RBFmdl:
    '''
    Construct radial basis function interpolant for DD problem. 
    This is used to compute an initial iterate to solve the DD-NM-ROM optimization problem. 
    
    fields:
    n_sub: number of subdomains in DD problem
    intr:  list of RBF interpolants for interior states. intr[j] corresponds to jth subdomain
    intf:  list of RBF interpolants for interface states. intf[j] corresponds to jth subdomain
    '''
    def __init__(self, mdl, params, interior, interface, neighbors=None, smoothing=0.0,
                 kernel='thin_plate_spline', epsilon=None, degree=None):
        '''
        inputs: 
        mdl:  DD_NM_ROM class
        params: (N, p)      input parameters for RBF interpolant
        interior:   list of interior-state snapshots associated to params for each subdomain
                    e.g. interior[j] = (N, nx_sub*ny_sub) interior snapshot array for subdomain j
        interface:  list of interface-state snapshots associated to params for each subdomain
                    e.g. interface[j] = (N, nx_sub*ny_sub) interface snapshot array for subdomain j
        neighbors: [optional] see scipy.interpolate.RBFInterpolator documentation. Default is None
        smoothing: [optional] see scipy.interpolate.RBFInterpolator documentation. Default is 0.0
        kernel:    [optional] see scipy.interpolate.RBFInterpolator documentation. Default is 'thin_plate_spline'
        epsilon:   [optional] see scipy.interpolate.RBFInterpolator documentation. Default is None
        degree:    [optional] see scipy.interpolate.RBFInterpolator documentation. Default is None
        '''
        
        self.n_sub = len(interior)
        n_params  = len(params)
        self.intr = []
        self.intf = []
        
        for j, s in enumerate(mdl.subdomain):
            S_intr = []
            S_intf = []
            
            for i in range(n_params):
                S_intr.append(s.en_intr(interior[j][i]))
                S_intf.append(s.en_intf(interface[j][i]))
                
            self.intr.append(RBFInterpolator(params, np.vstack(S_intr), neighbors=neighbors,
                                            smoothing=smoothing, kernel=kernel, epsilon=epsilon, degree=degree))
            self.intf.append(RBFInterpolator(params, np.vstack(S_intf), neighbors=neighbors, 
                                            smoothing=smoothing, kernel=kernel, epsilon=epsilon, degree=degree))
    
    def get_initial(self, mu, mdl):
        '''
        Computes initial iterate used for solving NM-ROM optimization subproblem. 
        
        inputs:
        mu:    (1, 2) array corresponding to parameters for the 2D Burgers problem.
        mdl:   DD_NM_ROM class 
        
        outputs:
        w0:    interior- and interface- state vector for initial guess to optimization problem 
        lam0:  initial guess for lagrange multipliers. computed using least-squares 
        '''
        w0 = []
        A  = []
        b  = []
        for j, s in enumerate(mdl.subdomain):
            w_intr = self.intr[j](mu).reshape(-1)
            w_intf = self.intf[j](mu).reshape(-1)
            
            w0.append(w_intr)
            w0.append(w_intf)
            
            res, jac, H, rhs, Ag, Adg = s.res_jac(w_intr, w_intf, np.zeros(mdl.n_constraints))
            
            A.append(Adg.T)
            b.append(jac.T[-s.intf_latent_dim:]@res)
        
        w0 = np.concatenate(w0)
        A  = np.vstack(A)
        b  = -np.concatenate(b)
        lam0 = la.lstsq(A, b)[0]
        return w0, lam0
    
def get_net_np_params(state_dict, mask_shape):
    '''
    Extracts weights and biases from the encoder or decoder state dict and converts to numpy/csr arrays.
    
    input:
    state_dict: PyTorch state dict for decoder
    mask: sparsity mask 
    
    output:
    W1: dense numpy array of weights for first layer
    b1: vector of biases for first layer
    W2: sparse csr matrix of weights in second layer

    '''
    b1   = state_dict['net.0.bias'].to('cpu').detach().numpy()
    
    if 'net.0.weight_mask' in state_dict:
        mask = state_dict['net.0.weight_mask'].to('cpu').detach().numpy()
        W1   = sp.csr_matrix(state_dict['net.0.weight_orig'].to('cpu').detach().numpy()*mask)
    elif 'net.0.weights' in state_dict:
        W1 = sp.csr_matrix((state_dict['net.0.weights'].to('cpu').detach().numpy(),
                   (state_dict['net.0.indices'][0].to('cpu').detach().numpy(), 
                    state_dict['net.0.indices'][1].to('cpu').detach().numpy())), 
                    shape=mask_shape)
    else: 
        W1 = state_dict['net.0.weight'].to('cpu').detach().numpy()
        
    if 'net.2.weight_mask' in state_dict:
        mask = state_dict['net.2.weight_mask'].to('cpu').detach().numpy()
        W2   = sp.csr_matrix(state_dict['net.2.weight_orig'].to('cpu').detach().numpy()*mask)
        
    elif 'net.2.weights' in state_dict:
        W2 = sp.csr_matrix((state_dict['net.2.weights'].to('cpu').detach().numpy(),
                   (state_dict['net.2.indices'][0].to('cpu').detach().numpy(), 
                    state_dict['net.2.indices'][1].to('cpu').detach().numpy())), 
                         mask_shape)
    else: 
        W2 = state_dict['net.2.weight'].to('cpu').detach().numpy()
    return W1, b1, W2

def compute_residual_bases(ddmdl, 
                  residuals,
                  ec=1e-8, 
                  nbasis=-1):
    '''
    Compute residual bases for each subdomain in the DD model. To be used in hyper-reduction.
    
    inputs: 
    ddmdl: instance of DD_model class
    residuals: list where residuals[i][j] = Newton residual at jth iteration correpsonding to ith parameter set
    ec: [optional] residual basis energy criterion. Default is 1e-8
    nbasis: [optional] number of basis residual basis vectors. Default is -1. See POD documentation.
    
    outputs:
    residual_bases: list where residual_bases[i] is residual basis for ith subdomain
    '''
    
    print('Computing residual bases for each subdomain...')
    
    # collect residual snapshots on each subdomain
    Nr = len(residuals)
    residual = []
    for s in ddmdl.subdomain:
        residual_i = []
        for j in range(Nr):
            residual_i.append(np.hstack([residuals[j][:, s.residual_ind], 
                                         residuals[j][:, s.residual_ind+ddmdl.nxy]]))
        residual.append(residual_i)

    # compute residual bases
    residual_bases  = list([])
    for i in range(ddmdl.n_sub):
        res_i, s = POD(np.vstack(residual[i]).T, ec=ec, n_basis=nbasis)
        residual_bases.append(res_i)
        print(f'residual_bases[{i}].shape={residual_bases[i].shape}')

    print('Bases computed!')
    
    return residual_bases
    
def get_hr_col_ind(row_ind, mat_list):
    '''
    Given HR row-nodes, get indices of nonzero column entries for each matrix in mat_list.
    
    input:
    row_ind: array/list of row indices 
    mat_list: list of matrices to get column indices from
    
    output:
    col_ind: column indices
    '''
    
    mat_list = [mat_list] if not isinstance(mat_list, list) else mat_list
    
    col_ind = set()
    for M in mat_list:
        M = sp.coo_matrix(M)
        for row in row_ind:
            col_ind = col_ind.union(set(M.col[M.row==row]))
    col_ind = np.sort(np.array(list(col_ind)))
    
    return col_ind 

def find_subnet(nodes, W1, b1, W2, scale, ref):
    '''
    Finds subnet for decoder.
    
    input:
    nodes: hyper reduction indices for output layer
    W1: weights for first layer
    b1: biases for first layer
    W2: weights for second layer
    scale: scaling vector for de-normalization 
    ref: shifting vector for de-normalization 
    
    output:
    W1: hyper-reduced weights for first layer
    b1: hyper-reduced biases for first layer
    W2: hyper-reduced weights for second layer
    scale: hyper-reduced scaling vector for de-normalization 
    ref: hyper-reduced shifting vector for de-normalization 
    '''
    ind = get_hr_col_ind(nodes, W2)
    return W1[ind], b1[ind], W2[np.ix_(nodes, ind)], scale[nodes], ref[nodes], ind
    
def sp_diag(vec):
        '''
        Helper function to compute sparse diagonal matrix.
        
        input:
        vec: vector to compute sparse diagonal matrix of
        
        output:
        D: sparse diagonal matrix with diagonal vec
        '''
        return sp.spdiags(vec, 0, vec.size, vec.size)

def sigmoid(z):
    '''
    Computes sigmoid activation, including first and second derivatives
    
    input:
    z: input vector
    
    output:
    a: output after activation
    da: derivative of activation output
    '''
    
    emz = np.exp(-z)
    a   = 1.0/(1.0+emz)
    da  = a*a*emz
#     d2a = a*da*(emz-1.0)
    return a, da #, d2a
def swish(z):
    '''
    Computes swish activation, including first and second derivatives
    
    input:
    z: input vector
    
    output:
    a: output after activation
    da: derivative of activation output
    '''
    
    emz = np.exp(-z)
    
    sig   = 1.0/(1.0+emz)
    dsig  = sig*sig*emz
#     d2sig = sig*dsig*(emz-1.0)
    
    a   = z*sig
    da  = sig + z*dsig
#     d2a = 2*dsig+z*d2sig
    return a, da #, d2a

# compute face-splitting product
def face_splitting(A, B):
    '''
    Compute face-splitting product of matrices A and B. Useful in the following identity:
        for matrices A, B and vectors x, y
        hadamard(Ax, By) = face_splitting(A, B) @ kron(x, y).
        
    It is related to the Khatri-Rao product by
        face_splitting(A, B).T = khatri_rao(A.T, B.T)
    and is equivalent to the row-wise kronecker product of A and B.
    
    inputs:
        A: (m, n) array
        B: (m, p) array
    outputs:
        C: (m, np) array 
    '''
    rows = []
    for j in range(A.shape[0]):
        rows.append(sp.kron(A[j].reshape(1, -1), B[j].reshape(1, -1)))
    return sp.vstack(rows)

class subdomain_NM_ROM:
    '''
    Class for generating a non-hyper-reduced subdomain of the DD NM-ROM for the 2D steady-state Burgers' Equation with Dirichlet BC.
    
    inputs: 
    subdomain: subdomain class of full-order DD model
    intr_dict: dictionary for interior states with fields
                decoder:    state_dict with decoder parameters
                latent_dim: latent dimension for decoder
                scale:      scaling vector for normalizing data
                ref:        reference vector for shifting data
    intf_dict: dictionary for interface states with fields
                decoder:    state_dict with decoder parameters
                latent_dim: latent dimension for decoder
                scale:      scaling vector for normalizing data
                ref:        reference vector for shifting data
    constraint_mult: multiplier for constraint matrix
    
    methods:
    en_intr:  encoder function for interior state
    en_intf:  encoder function for interface state
    de_intr : decoder function for interior state
    de_intf : decoder function for interface state 
    res_jac: compute residual its jacobian on the subdomain
    '''
    
    def __init__(self, subdomain, intr_dict, intf_dict, constraint_mult):
        self.in_ports  = subdomain.in_ports
        
        self.n_residual  = subdomain.n_residual
        self.n_interior  = subdomain.n_interior
        self.n_interface = subdomain.n_interface
        
        self.interior_ind  = subdomain.interior_ind
        self.interface_ind = subdomain.interface_ind
        
        self.intr_latent_dim = intr_dict['latent_dim']
        self.intf_latent_dim = intf_dict['latent_dim']
        
        if intr_dict['act_type'] == 'Sigmoid':
            self.intr_act_fun  = sigmoid
        elif intr_dict['act_type'] == 'Swish':
            self.intr_act_fun  = swish
            
        if intf_dict['act_type'] == 'Sigmoid':
            self.intf_act_fun  = sigmoid
        elif intr_dict['act_type'] == 'Swish':
            self.intf_act_fun  = swish
            
        # interior encoder weights and biases
        self.en_mask_intr = intr_dict['mask'].T
        self.en_W1_intr, self.en_b1_intr, self.en_W2_intr = get_net_np_params(intr_dict['encoder'], self.en_mask_intr.shape)
        self.scale_intr = intr_dict['scale'].to('cpu').detach().numpy()
        self.ref_intr   = intr_dict['ref'].to('cpu').detach().numpy()
        
        # interior decoder weights and biases
        self.de_mask_intr = intr_dict['mask']
        self.de_W1_intr, self.de_b1_intr, de_W2_intr= get_net_np_params(intr_dict['decoder'], self.de_mask_intr.shape)
        self.de_W2_intr  = sp_diag(self.scale_intr)@de_W2_intr
        
        # interface encoder weights and biases
        self.en_mask_intf = intf_dict['mask'].T
        self.en_W1_intf, self.en_b1_intf, self.en_W2_intf = get_net_np_params(intf_dict['encoder'], self.en_mask_intf.shape)
        self.scale_intf = intf_dict['scale'].to('cpu').detach().numpy()
        self.ref_intf   = intf_dict['ref'].to('cpu').detach().numpy()
        
        # interface decoder weights and biases
        self.de_mask_intf = intf_dict['mask']
        self.de_W1_intf, self.de_b1_intf, de_W2_intf = get_net_np_params(intf_dict['decoder'], self.de_mask_intf.shape)
        self.de_W2_intf  = sp_diag(self.scale_intf)@de_W2_intf
        
        # matrices for computing interior state residual
        self.Bx_interior  = copy(subdomain.Bx_interior)
        self.By_interior  = copy(subdomain.By_interior)
        self.Cx_interior  = copy(subdomain.Cx_interior)
        self.Cy_interior  = copy(subdomain.Cy_interior)
        
        # matrices for computing interface state residual
        self.Bx_interface = copy(subdomain.Bx_interface)
        self.By_interface = copy(subdomain.By_interface)
        self.Cx_interface = copy(subdomain.Cx_interface)
        self.Cy_interface = copy(subdomain.Cy_interface)
        
        self.b_ux1 = copy(subdomain.b_ux1)
        self.b_uy1 = copy(subdomain.b_uy1)
        self.b_ux2 = copy(subdomain.b_ux2)
        self.b_uy2 = copy(subdomain.b_uy2)
        
        self.b_vx1 = copy(subdomain.b_vx1)
        self.b_vy1 = copy(subdomain.b_vy1)
        self.b_vx2 = copy(subdomain.b_vx2)
        self.b_vy2 = copy(subdomain.b_vy2)
        
        # inclusion operators taking interior/interface nodes to residual nodes
        self.I_interior = copy(subdomain.I_interior)
        self.I_interface = copy(subdomain.I_interface)
        
        self.constraint_mat = constraint_mult@subdomain.constraint_mat #sp.block_diag([subdomain.constraint_mat,
                                                            # subdomain.constraint_mat])
        self.W2TAT = (self.constraint_mat@self.de_W2_intf).T
        
    def en_intr(self, w):
        '''
        Computes output of encoder for interior states
        
        input:
            w: (latent_dim,) vector
        output: 
            out: output of encoder 
        '''
        
        z1  = self.en_W1_intr@((w-self.ref_intr)/self.scale_intr) + self.en_b1_intr
#         a1, da1, d2a1 = self.intr_act_fun(z1)
        a1, da1 = self.intr_act_fun(z1)
        
        out = self.en_W2_intr@a1
        return out
    
    def en_intf(self, w):
        '''
        Computes output of encoder for interface states
        
        input:
            w: (latent_dim,) vector
        output: 
            out: output of encoder 
        '''
        
        z1  = self.en_W1_intf@((w-self.ref_intf)/self.scale_intf) + self.en_b1_intf
#         a1, da1, d2a1 = self.intf_act_fun(z1)
        a1, da1 = self.intf_act_fun(z1)

        out = self.en_W2_intf@a1
        return out
    
    def de_intr(self, w):
        '''
        Computes output of decoder and jacobian with respect to inputs for interior states
        
        input:
            w: (latent_dim,) vector
        output: 
            out: output of decoder
            jac: jacobian of output 
        '''
        z1  = self.de_W1_intr@w + self.de_b1_intr
#         a1, da1, d2a1 = self.intr_act_fun(z1)
        a1, da1 = self.intr_act_fun(z1)
        out = self.de_W2_intr@a1 + self.ref_intr
        jac = self.de_W2_intr@sp_diag(da1)@self.de_W1_intr
        return out, jac #, d2a1
    
    def de_intf(self, w):
        '''
        Computes output of decoder and jacobian with respect to inputs for interface states
        
        input:
            w: (latent_dim,) vector
        output: 
            out: output of decoder
            jac: jacobian of output 
        '''
        z1  = self.de_W1_intf@w + self.de_b1_intf
#         a1, da1, d2a1 = self.intf_act_fun(z1)
        a1, da1 = self.intf_act_fun(z1)
        
        out = self.de_W2_intf@a1 + self.ref_intf
        jac = self.de_W2_intf@sp_diag(da1)@self.de_W1_intf
        return out, jac #, d2a1
    
    def res_jac(self, w_intr, w_intf, lam):
        '''
        Compute residual and its jacobian on subdomain.
        
        inputs:
        w_intr: (n_interior,) vector - reduced interior state
        w_intf: (n_interface,) vector - reduced interface state
        lam:    (n_constraints,) vector of lagrange multipliers
        
        outputs:
        res: (nz,) vector - residual on subdomain
        jac: (nz, n_interior + n_interface) array - jacobian of residual w.r.t. (w_intr, w_intf)
        H:   Hessian submatrix for SQP solver
        rhs: RHS block vector in SQP solver
        Ag : constraint matrix times output of interface decoder
        Adg: constraint matrix times jacobian of interface decoder
        '''
        
#         uv_intr, de_intr_jac, d2a1_intr = self.de_intr(w_intr)
        uv_intr, de_intr_jac = self.de_intr(w_intr)
        u_intr  = uv_intr[:self.n_interior]
        v_intr  = uv_intr[self.n_interior:]
        
#         uv_intf, de_intf_jac, d2a1_intf = self.de_intf(w_intf)
        uv_intf, de_intf_jac = self.de_intf(w_intf)
        u_intf  = uv_intf[:self.n_interface]
        v_intf  = uv_intf[self.n_interface:]
        
        u_res = self.I_interior@u_intr + self.I_interface@u_intf
        v_res = self.I_interior@v_intr + self.I_interface@v_intf
        
        # compute matrix-vector products on interior and interface
        Bxu = self.Bx_interior@u_intr + self.Bx_interface@u_intf
        Byu = self.By_interior@u_intr + self.By_interface@u_intf
        Cxu = self.Cx_interior@u_intr + self.Cx_interface@u_intf
        Cyu = self.Cy_interior@u_intr + self.Cy_interface@u_intf
        
        Bxv = self.Bx_interior@v_intr + self.Bx_interface@v_intf
        Byv = self.By_interior@v_intr + self.By_interface@v_intf
        Cxv = self.Cx_interior@v_intr + self.Cx_interface@v_intf
        Cyv = self.Cy_interior@v_intr + self.Cy_interface@v_intf
        
        # computes u and v residuals
        res_u = u_res*(Bxu - self.b_ux1) + v_res*(Byu - self.b_uy1) \
              + Cxu + self.b_ux2 + Cyu + self.b_uy2
        res_v = u_res*(Bxv - self.b_vx1) + v_res*(Byv - self.b_vy1) \
              + Cxv + self.b_vx2 + Cyv + self.b_vy2
        res = np.concatenate((res_u, res_v))
        
        # precompute reused quantities
        diagU = sp_diag(u_res)
        diagV = sp_diag(v_res)
        diagU_Bx_intr = diagU@self.Bx_interior
        diagV_By_intr = diagV@self.By_interior
        diagU_Bx_intf = diagU@self.Bx_interface
        diagV_By_intf = diagV@self.By_interface
        
        # jacobian with respect to interior states
        jac_intr_uu = sp_diag(Bxu-self.b_ux1)@self.I_interior \
                          + diagU_Bx_intr + diagV_By_intr + self.Cx_interior + self.Cy_interior 
        jac_intr_uv = sp_diag(Byu-self.b_uy1)@self.I_interior
        jac_intr_vu = sp_diag(Bxv-self.b_vx1)@self.I_interior
        jac_intr_vv = diagU_Bx_intr \
                          + sp_diag(Byv-self.b_vx1)@self.I_interior \
                          + diagV_By_intr + self.Cx_interior + self.Cy_interior
        jac_intr = sp.bmat([[jac_intr_uu, jac_intr_uv],[jac_intr_vu, jac_intr_vv]])@de_intr_jac
        
        # jacobian with respect to interface states
        jac_intf_uu = sp_diag(Bxu-self.b_ux1)@self.I_interface \
                          + diagU_Bx_intf + diagV_By_intf + self.Cx_interface + self.Cy_interface
        jac_intf_uv = sp_diag(Byu-self.b_uy1)@self.I_interface
        jac_intf_vu = sp_diag(Bxv-self.b_vx1)@self.I_interface
        jac_intf_vv = diagU_Bx_intf + sp_diag(Byv-self.b_vx1)@self.I_interface \
                          + diagV_By_intf + self.Cx_interface + self.Cy_interface
        jac_intf = sp.bmat([[jac_intf_uu, jac_intf_uv],[jac_intf_vu, jac_intf_vv]])@de_intf_jac
        
        # compute terms needed for SQP solver 
        Ag  = self.constraint_mat@uv_intf
        Adg = self.constraint_mat@de_intf_jac
        
        jac = np.hstack([jac_intr, jac_intf])
        H   = jac.T@jac
        rhs = np.concatenate([jac_intr.T@res, jac_intf.T@res+Adg.T@lam])
        
#         JacTRes_intr = jac_intr.T@res
#         JacTRes_intf = jac_intf.T@res
        
#         rhs = np.concatenate([de_intr_jac.T@JacTRes_intr, de_intf_jac.T@JacTRes_intf+Adg.T@lam])
        
#         jac = np.hstack([jac_intr@de_intr_jac, jac_intf@de_intf_jac])
#         H   = jac.T@jac
        
#         G_intr = self.de_W1_intr.T@sp_diag((self.de_W2_intr.T@JacTRes_intr)*d2a1_intr)@self.de_W1_intr
#         G_intf = self.de_W1_intf.T@sp_diag((self.W2TAT@lam+self.de_W2_intf.T@JacTRes_intf)*d2a1_intf)@self.de_W1_intf
# #         G_intf = self.de_W1_intf.T@sp_diag((self.W2TAT@lam)*d2a1_intf)@self.de_W1_intf

#         H[:self.intr_latent_dim, :self.intr_latent_dim] += G_intr
#         H[-self.intf_latent_dim:, -self.intf_latent_dim:] += G_intf
    
        return res, jac, H, rhs, Ag, Adg

class subdomain_NM_ROM_HR:
    '''
    Class for generating a hyper-reduced subdomain of the DD NM-ROM for the 2D steady-state Burgers' Equation with Dirichlet BC.
    
    inputs: 
    subdomain: subdomain class of full-order DD model
    intr_dict: dictionary for interior states with fields
                decoder:    state_dict with decoder parameters
                latent_dim: latent dimension for decoder
                scale:      scaling vector for normalizing data
                ref:        reference vector for shifting data
    intf_dict: dictionary for interface states with fields
                decoder:    state_dict with decoder parameters
                latent_dim: latent dimension for decoder
                scale:      scaling vector for normalizing data
                ref:        reference vector for shifting data
    residual_basis: POD basis for residuals. used for computing hyper-reduction nodes.
    hr_type:    'collocation' or 'gappy_POD'. currently only 'collocation' has been implemented
    constraint_mult: multiplier for constraint matrix
    nz: integer of desired number of sample nodes
    ncol: number of working columns of residual_basis
    n_corners: number of interface nodes to include in sample nodes
    
    methods:
    en_intr:  encoder function for interior state
    en_intf:  encoder function for interface state
    de_intr : decoder function for interior state
    de_intf : decoder function for interface state 
    res_jac: compute residual its jacobian on the subdomain
    '''
    def __init__(self, subdomain,
                 intr_dict, 
                 intf_dict, 
                 residual_basis, 
                 hr_type, 
                 constraint_mult, 
                 nz, ncol, n_corners=5):
        
        self.in_ports  = subdomain.in_ports
        
        # number of nodes on residual subdomain, interior states and interface states
        self.n_residual  = subdomain.n_residual
        self.n_interior  = subdomain.n_interior
        self.n_interface = subdomain.n_interface
        
        # stores indices of interior/interface states relative to full domain 
        self.interior_ind  = subdomain.interior_ind
        self.interface_ind = subdomain.interface_ind
        
        self.intr_latent_dim = intr_dict['latent_dim']
        self.intf_latent_dim = intf_dict['latent_dim']
        
        if intr_dict['act_type'] == 'Sigmoid':
            self.intr_act_fun  = sigmoid
        elif intr_dict['act_type'] == 'Swish':
            self.intr_act_fun  = swish
            
        if intf_dict['act_type'] == 'Sigmoid':
            self.intf_act_fun  = sigmoid
        elif intf_dict['act_type'] == 'Swish':
            self.intf_act_fun  = swish
            
        # select residual nodes for hyper reduction
        self.hr_ind = select_sample_nodes(subdomain, residual_basis, nz, ncol, n_corners=n_corners)
        self.hr_ind_u_res = self.hr_ind[self.hr_ind < subdomain.n_residual]
        self.hr_ind_v_res = self.hr_ind[self.hr_ind >= subdomain.n_residual]-subdomain.n_residual
        
        # residual weighting matrix corresponding to HR type
        if hr_type == 'gappy_POD':
            self.res_weight   = la.pinv(self.residual_basis[self.hr_ind])
        elif hr_type == 'collocation':
            self.res_weight = sp.eye(self.hr_ind.size)
            
        # finds nodes for interior and interface states
        self.hr_ind_u_intr = get_hr_col_ind(self.hr_ind_u_res, [subdomain.I_interior, 
                                                                subdomain.Bx_interior, 
                                                                subdomain.By_interior, 
                                                                subdomain.Cx_interior, 
                                                                subdomain.Cy_interior])
        self.hr_ind_v_intr = get_hr_col_ind(self.hr_ind_v_res, [subdomain.I_interior, 
                                                                subdomain.Bx_interior, 
                                                                subdomain.By_interior, 
                                                                subdomain.Cx_interior, 
                                                                subdomain.Cy_interior])
        self.hr_ind_u_intf = get_hr_col_ind(self.hr_ind_u_res, [subdomain.I_interface, 
                                                                subdomain.Bx_interface, 
                                                                subdomain.By_interface, 
                                                                subdomain.Cx_interface, 
                                                                subdomain.Cy_interface])
        self.hr_ind_v_intf = get_hr_col_ind(self.hr_ind_v_res, [subdomain.I_interface, 
                                                                subdomain.Bx_interface, 
                                                                subdomain.By_interface, 
                                                                subdomain.Cx_interface, 
                                                                subdomain.Cy_interface])
        self.hr_ind_uv_intr = np.concatenate([self.hr_ind_u_intr, self.hr_ind_v_intr+self.n_interior]).astype(int)
        self.hr_ind_uv_intf = np.concatenate([self.hr_ind_u_intf, self.hr_ind_v_intf+self.n_interface]).astype(int)
        
        # saves submatrices corresponding to remaining nodes
        submatrix_indices = np.ix_(self.hr_ind_u_res, self.hr_ind_u_intr)
        self.Bx_intr_u = subdomain.Bx_interior[submatrix_indices]
        self.By_intr_u = subdomain.By_interior[submatrix_indices]
        self.Cx_intr_u = subdomain.Cx_interior[submatrix_indices]
        self.Cy_intr_u = subdomain.Cy_interior[submatrix_indices]
        
        submatrix_indices = np.ix_(self.hr_ind_v_res, self.hr_ind_v_intr)
        self.Bx_intr_v = subdomain.Bx_interior[submatrix_indices]
        self.By_intr_v = subdomain.By_interior[submatrix_indices]
        self.Cx_intr_v = subdomain.Cx_interior[submatrix_indices]
        self.Cy_intr_v = subdomain.Cy_interior[submatrix_indices]
        
        submatrix_indices = np.ix_(self.hr_ind_u_res, self.hr_ind_u_intf)
        self.Bx_intf_u = subdomain.Bx_interface[submatrix_indices]
        self.By_intf_u = subdomain.By_interface[submatrix_indices]
        self.Cx_intf_u = subdomain.Cx_interface[submatrix_indices]
        self.Cy_intf_u = subdomain.Cy_interface[submatrix_indices]
        
        submatrix_indices = np.ix_(self.hr_ind_v_res, self.hr_ind_v_intf)
        self.Bx_intf_v = subdomain.Bx_interface[submatrix_indices]
        self.By_intf_v = subdomain.By_interface[submatrix_indices]
        self.Cx_intf_v = subdomain.Cx_interface[submatrix_indices]
        self.Cy_intf_v = subdomain.Cy_interface[submatrix_indices]
        
        # hyper-reduced inclusion matrices
        self.uu_intr_incl = subdomain.I_interior[np.ix_(self.hr_ind_u_res, self.hr_ind_u_intr)]
        self.uv_intr_incl = subdomain.I_interior[np.ix_(self.hr_ind_u_res, self.hr_ind_v_intr)]
        self.uu_intf_incl = subdomain.I_interface[np.ix_(self.hr_ind_u_res, self.hr_ind_u_intf)]
        self.uv_intf_incl = subdomain.I_interface[np.ix_(self.hr_ind_u_res, self.hr_ind_v_intf)]
        
        self.vu_intr_incl = subdomain.I_interior[np.ix_(self.hr_ind_v_res, self.hr_ind_u_intr)]
        self.vv_intr_incl = subdomain.I_interior[np.ix_(self.hr_ind_v_res, self.hr_ind_v_intr)]
        self.vu_intf_incl = subdomain.I_interface[np.ix_(self.hr_ind_v_res, self.hr_ind_u_intf)]
        self.vv_intf_incl = subdomain.I_interface[np.ix_(self.hr_ind_v_res, self.hr_ind_v_intf)]
        
        self.b_ux1 = subdomain.b_ux1[self.hr_ind_u_res]
        self.b_uy1 = subdomain.b_uy1[self.hr_ind_u_res]
        self.b_ux2 = subdomain.b_ux2[self.hr_ind_u_res]
        self.b_uy2 = subdomain.b_uy2[self.hr_ind_u_res]
        
        self.b_vx1 = subdomain.b_vx1[self.hr_ind_v_res]
        self.b_vy1 = subdomain.b_vy1[self.hr_ind_v_res]
        self.b_vx2 = subdomain.b_vx2[self.hr_ind_v_res]
        self.b_vy2 = subdomain.b_vy2[self.hr_ind_v_res]
        
        # interior encoder weights and biases
        self.en_mask_intr = intr_dict['mask'].T
        self.en_W1_intr, self.en_b1_intr, self.en_W2_intr = get_net_np_params(intr_dict['encoder'], self.en_mask_intr.shape)
        self.scale_intr = intr_dict['scale'].to('cpu').detach().numpy()
        self.ref_intr   = intr_dict['ref'].to('cpu').detach().numpy()
        
        # interior decoder weights and biases
        self.de_mask_intr = intr_dict['mask']
        self.de_W1_intr, self.de_b1_intr, de_W2_intr = get_net_np_params(intr_dict['decoder'], self.de_mask_intr.shape)
        self.de_W2_intr  = sp_diag(self.scale_intr)@de_W2_intr
        
        # interface encoder weights and biases
        self.en_mask_intf = intf_dict['mask'].T
        self.en_W1_intf, self.en_b1_intf, self.en_W2_intf = get_net_np_params(intf_dict['encoder'], self.en_mask_intf.shape)
        self.scale_intf = intf_dict['scale'].to('cpu').detach().numpy()
        self.ref_intf   = intf_dict['ref'].to('cpu').detach().numpy()
        
        # interface decoder weights and biases
        self.de_mask_intf = intf_dict['mask']
        self.de_W1_intf, self.de_b1_intf, de_W2_intf = get_net_np_params(intf_dict['decoder'], self.de_mask_intf.shape)
        self.de_W2_intf  = sp_diag(self.scale_intf)@de_W2_intf
        
        # interior decoder subnet weights and biases
        self.de_W1_intr_hr, self.de_b1_intr_hr, de_W2_intr_hr, scale_intr_hr, self.ref_intr_hr, ind = find_subnet(self.hr_ind_uv_intr, 
                                                                                                   self.de_W1_intr, 
                                                                                                   self.de_b1_intr, 
                                                                                                   de_W2_intr, 
                                                                                                   self.scale_intr, 
                                                                                                   self.ref_intr)
        self.de_W2_intr_hr = sp_diag(scale_intr_hr)@de_W2_intr_hr
        
        
        # interface decoder subnet weights and biases
        self.de_W1_intf_hr, self.de_b1_intf_hr, de_W2_intf_hr, scale_intf_hr, self.ref_intf_hr, ind = find_subnet(self.hr_ind_uv_intf, 
                                                                                                    self.de_W1_intf, 
                                                                                                    self.de_b1_intf, 
                                                                                                    de_W2_intf, 
                                                                                                    self.scale_intf, 
                                                                                                    self.ref_intf)
        self.de_intf_inner_hr_nodes = ind
        self.de_W2_intf_hr = sp_diag(scale_intf_hr)@de_W2_intf_hr
        
        self.constraint_mat = constraint_mult@subdomain.constraint_mat #sp.block_diag([subdomain.constraint_mat,
                                              #               subdomain.constraint_mat])
        
        self.W2TAT = (self.constraint_mat@self.de_W2_intf).T#[self.de_intf_inner_hr_nodes]
    
    def en_intr(self, w):
        '''
        Computes output of encoder for interior states
        
        input:
            w: (latent_dim,) vector
        output: 
            out: output of encoder 
        '''
        
        z1  = self.en_W1_intr@((w-self.ref_intr)/self.scale_intr) + self.en_b1_intr
#         a1, da1, d2a1 = self.intr_act_fun(z1)
        a1, da1 = self.intr_act_fun(z1)

        out = self.en_W2_intr@a1
        return out
    
    def en_intf(self, w):
        '''
        Computes output of encoder for interface states
        
        input:
            w: (latent_dim,) vector
        output: 
            out: output of encoder 
        '''
        
        z1  = self.en_W1_intf@((w-self.ref_intf)/self.scale_intf) + self.en_b1_intf
#         a1, da1, d2a1 = self.intf_act_fun(z1)
        a1, da1 = self.intf_act_fun(z1)
        out = self.en_W2_intf@a1
        return out
    
    def de_intr(self, w):
        '''
        Computes output of decoder and jacobian with respect to inputs for interior states
        
        input:
            w: (latent_dim,) vector
        output: 
            out: output of decoder
            jac: jacobian of output 
        '''
        z1  = self.de_W1_intr@w + self.de_b1_intr
#         a1, da1, d2a1 = self.intr_act_fun(z1)
        a1, da1 = self.intr_act_fun(z1)
        out = self.de_W2_intr@a1 + self.ref_intr
        jac = self.de_W2_intr@sp_diag(da1)@self.de_W1_intr
        return out, jac #, d2a1
    
    def de_intr_hr(self, w):
        '''
        Computes hyper-reduced output of decoder and jacobian with respect to inputs for interior states
        
        input:
            w: (latent_dim,) vector
        output: 
            out: output of decoder
            jac: jacobian of output 
            d2a1: 2nd derivative of activation function in first layer
        '''
        z1  = self.de_W1_intr_hr@w + self.de_b1_intr_hr
#         a1, da1, d2a1 = self.intr_act_fun(z1)
        a1, da1 = self.intr_act_fun(z1)
        out = self.de_W2_intr_hr@a1 + self.ref_intr_hr
        jac = self.de_W2_intr_hr@sp_diag(da1)@self.de_W1_intr_hr
        return out, jac #, d2a1
    
    def de_intf(self, w):
        '''
        Computes output of decoder and jacobian with respect to inputs for interface states
        
        input:
            w: (latent_dim,) vector
        output: 
            out: output of decoder
            jac: jacobian of output 
            d2a1: 2nd derivative of activation function in first layer
        '''
        z1  = self.de_W1_intf@w + self.de_b1_intf
#         a1, da1, d2a1 = self.intf_act_fun(z1)
        a1, da1 = self.intf_act_fun(z1)
        out = self.de_W2_intf@a1 + self.ref_intf
        jac = self.de_W2_intf@sp_diag(da1)@self.de_W1_intf
        return out, jac #, d2a1
    
#     def de_intf_hr(self, w):
#         '''
#         Computes hyper-reduced output of decoder and jacobian with respect to inputs for interface states
        
#         input:
#             w: (latent_dim,) vector
#         output: 
#             out: output of decoder
#             dout: jacobian of output 
#         '''
#         z1   = self.W1_intf_hr@w + self.b1_intf_hr
#         a1   = self.intf_act_fun(z1)
#         out  = self.W2_intf_hr@a1 + self.ref_intf_hr
#         dout = self.W2_intf_hr@sp_diag(self.intf_dact_fun(z1))@self.W1_intf_hr
#         return out, dout
    
    def res_jac(self, w_intr, w_intf, lam):
        '''
        Compute residual and its jacobian on subdomain.
        
        inputs:
        w_intr: (n_interior,) vector - reduced interior state
        w_intf: (n_interface,) vector - reduced interface state
        lam:    (n_constraints,) vector of lagrange multipliers
        
        outputs:
        res: (nz,) vector - residual on subdomain
        jac: (nz, n_interior + n_interface) array - jacobian of residual w.r.t. (w_intr, w_intf)
        H:   Hessian submatrix for SQP solver
        rhs: RHS block vector in SQP solver
        Ag : constraint matrix times output of interface decoder
        Adg: constraint matrix times jacobian of interface decoder
        '''
        
#         uv_intr, de_intr_jac, d2a1_intr = self.de_intr_hr(w_intr)
        uv_intr, de_intr_jac = self.de_intr_hr(w_intr)
        u_intr  = uv_intr[:self.hr_ind_u_intr.size]
        v_intr  = uv_intr[-self.hr_ind_v_intr.size:]
        
#         uv_intf, de_intf_jac, d2a1_intf = self.de_intf(w_intf)
        uv_intf, de_intf_jac = self.de_intf(w_intf)
#         d2a1_intf_hr = d2a1_intf[self.de_intf_inner_hr_nodes]
        uv_intf_hr = uv_intf[self.hr_ind_uv_intf]
        de_intf_jac_hr = de_intf_jac[self.hr_ind_uv_intf]
        u_intf  = uv_intf_hr[:self.hr_ind_u_intf.size]
        v_intf  = uv_intf_hr[self.hr_ind_u_intf.size:]
        
        uu_res = self.uu_intr_incl@u_intr + self.uu_intf_incl@u_intf
        uv_res = self.uv_intr_incl@v_intr + self.uv_intf_incl@v_intf
        vu_res = self.vu_intr_incl@u_intr + self.vu_intf_incl@u_intf
        vv_res = self.vv_intr_incl@v_intr + self.vv_intf_incl@v_intf
        
        # compute matrix-vector products on interior and interface
        Bxu = self.Bx_intr_u@u_intr + self.Bx_intf_u@u_intf
        Byu = self.By_intr_u@u_intr + self.By_intf_u@u_intf
        Cxu = self.Cx_intr_u@u_intr + self.Cx_intf_u@u_intf
        Cyu = self.Cy_intr_u@u_intr + self.Cy_intf_u@u_intf
        
        Bxv = self.Bx_intr_v@v_intr + self.Bx_intf_v@v_intf
        Byv = self.By_intr_v@v_intr + self.By_intf_v@v_intf
        Cxv = self.Cx_intr_v@v_intr + self.Cx_intf_v@v_intf
        Cyv = self.Cy_intr_v@v_intr + self.Cy_intf_v@v_intf
        
        # computes u and v residuals
        res_u = uu_res*(Bxu - self.b_ux1) + uv_res*(Byu - self.b_uy1) \
              + Cxu + self.b_ux2 + Cyu + self.b_uy2
        res_v = vu_res*(Bxv - self.b_vx1) + vv_res*(Byv - self.b_vy1) \
              + Cxv + self.b_vx2 + Cyv + self.b_vy2
        res = np.concatenate((res_u, res_v))
        
        # precompute reused quantities 
        diagUU = sp_diag(uu_res)
        diagUV = sp_diag(uv_res)
        diagVU = sp_diag(vu_res)
        diagVV = sp_diag(vv_res)
        
        # jacobian with respect to interior states
        jac_intr_uu = sp_diag(Bxu-self.b_ux1)@self.uu_intr_incl \
                      + diagUU@self.Bx_intr_u + diagUV@self.By_intr_u + self.Cx_intr_u + self.Cy_intr_u
        jac_intr_uv = sp_diag(Byu-self.b_uy1)@self.uv_intr_incl
        jac_intr_vu = sp_diag(Bxv-self.b_vx1)@self.vu_intr_incl
        jac_intr_vv = diagVU@self.Bx_intr_v \
                      + sp_diag(Byv-self.b_vx1)@self.vv_intr_incl \
                      + diagVV@self.By_intr_v + self.Cx_intr_v + self.Cy_intr_v
        jac_intr = sp.bmat([[jac_intr_uu, jac_intr_uv],[jac_intr_vu, jac_intr_vv]])@de_intr_jac
        
        # jacobian with respect to intface states
        jac_intf_uu = sp_diag(Bxu-self.b_ux1)@self.uu_intf_incl \
                      + diagUU@self.Bx_intf_u + diagUV@self.By_intf_u + self.Cx_intf_u + self.Cy_intf_u
        jac_intf_uv = sp_diag(Byu-self.b_uy1)@self.uv_intf_incl
        jac_intf_vu = sp_diag(Bxv-self.b_vx1)@self.vu_intf_incl
        jac_intf_vv = diagVU@self.Bx_intf_v \
                      + sp_diag(Byv-self.b_vx1)@self.vv_intf_incl \
                      + diagVV@self.By_intf_v + self.Cx_intf_v + self.Cy_intf_v
        jac_intf = sp.bmat([[jac_intf_uu, jac_intf_uv],[jac_intf_vu, jac_intf_vv]])@de_intf_jac_hr
        
        # compute terms needed for SQP solver 
        Ag  = self.constraint_mat@uv_intf
        Adg = self.constraint_mat@de_intf_jac
        
        jac = np.hstack([jac_intr, jac_intf])
        H   = jac.T@jac
        rhs = np.concatenate([jac_intr.T@res, jac_intf.T@res+Adg.T@lam])
        
#         JacTRes_intr = jac_intr.T@res
#         JacTRes_intf = jac_intf.T@res
        
#         rhs = np.concatenate([de_intr_jac.T@JacTRes_intr, de_intf_jac_hr.T@JacTRes_intf+Adg.T@lam])
        
#         jac = np.hstack([jac_intr@de_intr_jac, jac_intf@de_intf_jac_hr])
#         H   = jac.T@jac
        
#         G_intr = self.de_W1_intr_hr.T@sp_diag((self.de_W2_intr_hr.T@JacTRes_intr)*d2a1_intr)@self.de_W1_intr_hr
#         G_intf = self.de_W1_intf_hr.T@sp_diag((self.de_W2_intf_hr.T@JacTRes_intf)*d2a1_intf_hr)@self.de_W1_intf_hr 
#         G_intf += self.de_W1_intf.T@sp_diag((self.W2TAT@lam)*d2a1_intf)@self.de_W1_intf

#         H[:self.intr_latent_dim, :self.intr_latent_dim] += G_intr
#         H[-self.intf_latent_dim:, -self.intf_latent_dim:] += G_intf
        
        return res, jac, H, rhs, Ag, Adg
        
class DD_NM_ROM:
    '''
    Compute DD NM-ROM for the 2D steady-state Burgers' equation with Dirichlet BC.
    
    inputs:
    ddmdl: DD model class corresponding to full order DD model. 
    intr_net_list: list of paths to trained networks for interior states
    intf_net_list: list of paths to trained networks for interface states
    residual_bases: [optional] list of residual bases where residual_bases[i] is the residual basis for the ith subdomain
    hr: [optional] Boolean to specify if hyper reduction is applied. Default is False
    hr_type: [optional] Hyper-reduction type. Either 'gappy_POD' or 'collocation'. 
                Only takes effect if hr=True. Default is 'collocation'
    sample_ratio: [optional] ratio of number of hyper-reduction samples to residual basis size. Default is 2
    n_samples: [optional] specify number of hyper reduction sample nodes. 
                If n_samples is an array with length equal to the number of subdomains, then 
                n_samples[i] is the number of HR samples on the ith subdomain.
                
                If n_samples is a positive integer, then each subdomain has n_samples HR nodes. 
                
                Otherwise, the number of samples is determined by the sample ratio. 
                Default is -1. 
                
    n_corners: [optional] Number of interface nodes included in the HR sample nodes. 
                If n_corners is an array with length equal to the number of subdomains, then 
                n_corners[i] is the number of interface HR nodes on the ith subdomain.
                
                If n_corners is a positive integer, then each subdomain has n_corners interface HR nodes. 
                
                Otherwise, the number of interface HR nodes on each subdomain is determined by n_samples
                multiplied by the ratio of the number of interface nodes contained in the residual nodes 
                to the total number of residual nodes.
                
                Default is -1. 
    
    
    fields:
    n_constraints: number of (equality) constraints for NLP
    constraint_mult: matrix multiplier corresponding to constraint_type. 
                     if constraint_type == 'weak':
                         constraint_mult = (n_constraints, nA) Gaussian random matrix
                     elif constraint_type == 'strong':
                         constraint_mult = identity
    subdomain: list of subdomain_LS_ROM or subdomain_LS_ROM_HR classes corresponding to each subdomain, i.e.
               subdomain[i] = reduced subdomain class corresponding to subdomain [i]
               
    methods:
    FJac: computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver. 
    solve: solves for the reduced states of the DD NM-ROM using the Lagrange-Newton-SQP method. 
    '''
    def __init__(self, ddmdl, 
                 intr_net_list,
                 intf_net_list,
                 residual_bases=None, 
                 hr=False, 
                 hr_type='collocation',
                 sample_ratio=2,
                 n_samples=-1,
                 n_corners=-1, 
                 constraint_type='weak', 
                 n_constraints=1,
                 seed=None):
        self.hr  = hr
        self.nxy = ddmdl.nxy
        self.n_sub = ddmdl.n_sub
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if constraint_type=='weak': 
            self.n_constraints = n_constraints
            rng = np.random.default_rng(seed)
            self.constraint_mult = rng.standard_normal(size=(self.n_constraints, ddmdl.n_constraints))
        else: 
            self.n_constraints = ddmdl.n_constraints
            self.constraint_mult = np.eye(self.n_constraints)
        
        self.subdomain=list([])
        # generate subdomain models
        if hr:
            if residual_bases == None:
                print("Error: must pass residual_bases as input.")
                return 
            if hr_type in ['gappy_POD', 'collocation']:
                self.hr_type = hr_type
            else: 
                print("Error: hr_type must be 'gappy_POD' or 'collocation'.")
                return 
            
            # compute parameters for hyper reduction
            ncol   = np.array([rb.shape[1] for rb in residual_bases])             # number of working columns per subdomain
            nz_max = np.array([rb.shape[0] for rb in residual_bases])
            
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
            n_corners_max = nz_max - np.array([2*s.n_interior for s in ddmdl.subdomain])
            if isinstance(n_corners, int):
                if n_corners > 0:
                    n_corners = n_corners*np.ones(self.n_sub)
                else:
                    n_corners = np.round(nz*n_corners_max/nz_max)
            n_corners = np.minimum(n_corners, n_corners_max)
            
            # generate subdomain models
            for i, s in enumerate(ddmdl.subdomain):
                intr_dict = torch.load(intr_net_list[i], map_location=self.device)
                intf_dict = torch.load(intf_net_list[i], map_location=self.device)
                
                self.subdomain.append(subdomain_NM_ROM_HR(s, intr_dict, intf_dict,
                                                          residual_bases[i],
                                                          self.hr_type,
                                                          self.constraint_mult,
                                                          int(nz[i]),
                                                          int(ncol[i]), 
                                                          n_corners=int(n_corners[i])))
        else: 
            for i, s in enumerate(ddmdl.subdomain):
                intr_dict = torch.load(intr_net_list[i], map_location=self.device)
                intf_dict = torch.load(intf_net_list[i], map_location=self.device)
                self.subdomain.append(subdomain_NM_ROM(s, intr_dict, intf_dict, self.constraint_mult))
                                  
    # compute RHS and jacobian for SQP solver
    def FJac(self, w):
        '''
        Computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver. 
        
        inputs: 
        w: vector of all interior and interface states for each subdomain 
           and the lagrange multipliers lam in the order
            [intr[0], intf[0], ..., intr[n_sub], intf[n_sub], lam]
            
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
            # extracts part of vector corresponding to subdomain interior/interface states
            interior_ind  = np.arange(s.intr_latent_dim)
            interface_ind = np.arange(s.intf_latent_dim)

            w_intr = w[interior_ind+shift]
            shift += s.intr_latent_dim

            w_intf = w[interface_ind+shift]
            shift += s.intf_latent_dim
            
            # computes residual, jacobian, and other quantities needed for KKT system
            res, jac, H, rhs, Ag, Adg = s.res_jac(w_intr, w_intf, lam)
        
            # RHS block for KKT system
            val.append(rhs)
            constraint_res += Ag
            
            H_list.append(H)
            A_list += [sp.csr_matrix((self.n_constraints, s.intr_latent_dim)), Adg]

        val.append(constraint_res)
        val = np.concatenate(val)
        H_block = sp.block_diag(H_list)
        A_block = sp.hstack(A_list)
        full_jac = sp.bmat([[H_block, A_block.T], [A_block, None]]).tocsr()
        return val, full_jac
    
    # solve SQP using Lagrange-Newton SQP
    def solve(self, w0, tol=1e-5, maxit=20, print_hist=False, rhs_hist=False):
        '''
        Solves for the reduced states of the DD-LS-ROM using the Lagrange-Newton-SQP algorithm. 
        
        inputs: 
        w0: initial interior/interface states and lagrange multipliers
        tol: [optional] stopping tolerance for Newton solver. Default is 1e-5
        maxit: [optional] max number of iterations for newton solver. Default is 20
        print_hist: [optional] Boolean to print iteration history for Newton solver. Default is False
        rhs_hist: [optional] Boolean to return RHS vectors for each iteration. Default is False
        
        outputs:
        u_full: u state mapped to full domain
        v_full: v state mapped to full domain
        w_interior: list of interior states for each subdomain, i.e
                    w_interior[i] = interior state on subdomain i
        w_interface: list of interface states for each subdomain, i.e
                     w_interface[i] = interface state on subdomain i
        lam:    (n_constraints,) vector of lagrange multipliers
        runtime: solve time for Newton solver
        itr: number of iterations for Newton solver
        '''
        
        print('Starting Newton solver...')
        start = time()
        y, res_vecs, res_hist, step_hist, itr = newton_solve(self.FJac, w0, tol=tol, maxit=maxit, print_hist=print_hist)
        runtime = time()-start
        print(f'Newton solver terminated after {itr} iterations with residual {res_hist[-1]:1.4e}.')
        
        # assemble solution on full domain from DD-ROM solution
        u_full = np.zeros(self.nxy)
        v_full = np.zeros(self.nxy)
        w_interior = list([])
        w_interface = list([])
        lam = y[-self.n_constraints:]
        
        shift = 0
        for s in self.subdomain:
            w_intr = y[np.arange(s.intr_latent_dim)+shift]
            shift += s.intr_latent_dim
            w_intf = y[np.arange(s.intf_latent_dim)+shift]
            shift += s.intf_latent_dim
            
            uv_intr = s.de_intr(w_intr)[0]
            u_full[s.interior_ind] = uv_intr[0:s.n_interior]
            v_full[s.interior_ind] = uv_intr[s.n_interior:]
            
            uv_intf = s.de_intf(w_intf)[0]
            u_full[s.interface_ind] = uv_intf[0:s.n_interface]
            v_full[s.interface_ind] = uv_intf[s.n_interface:]
            
            w_interior.append(w_intr)
            w_interface.append(w_intf)
            
        if rhs_hist:
            return u_full, v_full, w_interior, w_interface, lam, runtime, itr, res_vecs
        else:
            return u_full, v_full, w_interior, w_interface, lam, runtime, itr
    
    # Compute error between ROM and FOM solutions
    def compute_error(self, w_intr, w_intf, u_intr, v_intr, u_intf, v_intf):
        '''
        Compute error between ROM and FOM DD solutions.
        
        inputs:
        w_intr: list of reduced interior states where
                   w_intr[i] = (self.subdomain[i].intr_latent_dim,) vector of ROM solution interior states
        w_intf: list of reduced interface states where
                   w_intf[i] = (self.subdomain[i].intf_latent_dim,) vector of ROM solution interface states
                   
        u_intr: list of FOM u interior states where
                   u_intr[i] = (ddmdl.subdomain[i].n_interior,) vector of FOM u solution interior states
        v_intr: list of FOM v interior states where
                   v_intr[i] = (ddmdl.subdomain[i].n_interior,) vector of FOM v solution interior states
        u_intf: list of FOM u interface states where
                   u_intf[i] = (ddmdl.subdomain[i].n_interface,) vector of FOM u solution interface states
        v_intf: list of FOM v interface states where
                   v_intf[i] = (ddmdl.subdomain[i].n_interface,) vector of FOM v solution interface states
                   
        output:
        error: square root of mean squared relative error on each subdomain
        '''
        error = 0.0
        for i, s in enumerate(self.subdomain):
            uv_intr = np.concatenate([u_intr[i], v_intr[i]])
            uv_intf = np.concatenate([u_intf[i], v_intf[i]])
            
            num = np.sum(np.square(uv_intr - s.de_intr(w_intr[i])[0])) +\
                  np.sum(np.square(uv_intf - s.de_intf(w_intf[i])[0]))

            den = np.sum(np.square(uv_intr)) +\
                  np.sum(np.square(uv_intf))

            error += num/den
        error = np.sqrt(error/self.n_sub)
        return error