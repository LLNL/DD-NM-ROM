import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from time import time
from utils.solvers import newton_solve
from utils.helpers import sp_diag, select_sample_nodes
from scipy.interpolate import RBFInterpolator
import dill as pickle
import sys
from copy import copy

class subdomain_rbf:
    '''
    Subdomain RBF interpolant.
    
    fields:
    interior:     RBF interpolant for interior states
    interface:    RBF interpolant for interface states
    
    methods:
    compute_guess: compute initial guess for states
    '''
    def __init__(self, subdomain, parameters, intr_snapshots, intf_snapshots, neighbors=None, smoothing=0.0,
                 kernel='thin_plate_spline', epsilon=None, degree=None):
        '''
        Initialize subdomain RBF interpolant class. 
        
        inputs:
        subdomain:       instance of subdomain NM-ROM class
        parameters:      parameters at which to train RBF interpolant
        intr_snapshots:  interior state snapshots
        intf_snapshots:  interface state snapshots
        neighbors:       [optional] see scipy.interpolate.RBFInterpolator documentation. Default is None
        smoothing:       [optional] see scipy.interpolate.RBFInterpolator documentation. Default is 0.0
        kernel:          [optional] see scipy.interpolate.RBFInterpolator documentation. Default is 'linear'
        epsilon:         [optional] see scipy.interpolate.RBFInterpolator documentation. Default is None
        degree:          [optional] see scipy.interpolate.RBFInterpolator documentation. Default is None
        '''
        encoded_intr = subdomain.interior.encoder.fwd(intr_snapshots)
        encoded_intf = subdomain.interface.encoder.fwd(intf_snapshots)
        
        self.interior = RBFInterpolator(parameters, encoded_intr, neighbors=neighbors,
                                        smoothing=smoothing, kernel=kernel, epsilon=epsilon, degree=degree)
        self.interface = RBFInterpolator(parameters, encoded_intf, neighbors=neighbors,
                                        smoothing=smoothing, kernel=kernel, epsilon=epsilon, degree=degree)
    
    def compute_guess(self, inputs, subdomain):
        '''
        Compute intial guess for subdomain interior and interface states.
        
        inputs:
        inputs:    parameter input for RBF interpolant
        subdomain: subdomain NM-ROM class instance
        
        outputs:
        w_intr:    interior state guess
        w_intf:    interface state guess
        '''
        w_intr = self.interior(inputs)
        w_intf = self.interface(inputs)
        
        return w_intr, w_intf
        
class dd_rbf:
    '''
    RBF interpolant for DD NM-ROM SQP solver initial guesses for each time step.
    
    fields: 
    parameters: (n_snaps, 2) array corresponding to training snapshots where parameters[i] = (mu, tk) 

    methods:
    compute_guess: compute intial guess of SQP solver
    subdomain:     list of subdomain_rbf classes corresponding to each subdomain 
    '''
    def __init__(self, ddnmrom, t_lim, nt, mu_list, snapshots, n_snaps=400, neighbors=None, smoothing=0.0,
                 kernel='thin_plate_spline', epsilon=None, degree=None):
        '''
        Initialize DD RBF interpolant class. 
        
        inputs:
        ddnmrom:   instance of DD NM-ROM class
        t_lim:     time domain of problem, where t_lim[0] = initial time, t_lim[1] = final time
        nt:        number of time steps
        mu_list:   list of mu parameters corresponding to snapshots used for interpolant training
        snapshots: array of training snapshots
        n_snaps:   [optional] number of snapshots to use for RBF training. Set to -1 to use all snapshots. Default is 400.
        neighbors: [optional] see scipy.interpolate.RBFInterpolator documentation. Default is None
        smoothing: [optional] see scipy.interpolate.RBFInterpolator documentation. Default is 0.0
        kernel:    [optional] see scipy.interpolate.RBFInterpolator documentation. Default is 'thin_plate_spline'
        epsilon:   [optional] see scipy.interpolate.RBFInterpolator documentation. Default is None
        degree:    [optional] see scipy.interpolate.RBFInterpolator documentation. Default is None
        '''
        
        # set parameters for RBF 
        self.parameters = np.vstack([np.kron(np.array(mu_list), np.ones(nt+1)),
                                np.kron(np.ones(len(mu_list)), np.linspace(t_lim[0], t_lim[1], nt+1))]).T
            
        # subsample snapshots
        if n_snaps < 0:
            n_snaps = snapshots.shape[1]
        else:
            idx = np.linspace(0, len(self.parameters), num=n_snaps, endpoint=False, dtype=int)
            self.parameters = self.parameters[idx]
            snapshots = snapshots[:, idx]
        
        # train RBF interpolants for each subdomain 
        self.subdomain = []
        for s in ddnmrom.subdomain:
            intr_snapshots = np.vstack([snapshots[s.interior.indices],
                                        snapshots[s.interior.indices+ddnmrom.nxy]]).T
            intf_snapshots = np.vstack([snapshots[s.interface.indices],
                                        snapshots[s.interface.indices+ddnmrom.nxy]]).T
            
            self.subdomain.append(subdomain_rbf(s, self.parameters, intr_snapshots, intf_snapshots,
                                                neighbors=neighbors, smoothing=smoothing, 
                                                kernel=kernel, epsilon=epsilon, degree=degree))
        self.n_sub = ddnmrom.n_sub
    
    def compute_guess(self, t_lim, nt, mu, ddnmrom):
        '''
        Compute initial guess for SQP solver.
        
        inputs: 
        t_lim:    time domain for problem
        nt:       number of time steps
        mu:       mu parameter
        ddnmrom:  instance of DD NM-ROM class
        
        outpus:
        w:        (nt, nD+nA) array of intial guesses for SQP solver at each time step
        runtiem:  runtime required to compute guess
        '''
        stimes  = np.zeros(self.n_sub)
        
        start   = time()
        w       = []
        tt      = np.linspace(t_lim[0], t_lim[1], nt+1)
        ht      = (t_lim[1]-t_lim[0])/nt
        inputs  = np.vstack([mu*np.ones(nt+1), tt]).T
        lam0    = np.zeros(ddnmrom.n_constraints)
        lam     = np.zeros((nt, ddnmrom.n_constraints))
        
        runtime = time()-start
        
        # compute guesses for state
        for j, s in enumerate(ddnmrom.subdomain):
            start  = time()
            w_intr, w_intf = self.subdomain[j].compute_guess(inputs, s)
            
            w.append(w_intr)
            w.append(w_intf)
            stimes[j] = time()-start
            
        runtime += stimes.max()
        start = time()
        
        # compute guess for Lagrange multipliers
        if ddnmrom.scaling >= 1:
            for k in range(nt):
                res, jac, H, rhs, Ag, Adg = s.res_jac(w_intr[k+1], w_intf[k+1], w_intr[k], w_intf[k], lam0, ht)
                A = Adg.toarray().T if sp.issparse(Adg) else Adg.T
                b = -rhs[-s.interface.romsize:]
                lam[k] = la.lstsq(A, b)[0]
            
        w = np.hstack(w)
        w = np.hstack([w[:nt], lam])
        runtime += time()-start
        
        return w, runtime

def get_net_np_params(state_dict, mask_shape):
    '''
    Extracts weights and biases from the encoder or decoder state dict and converts to numpy/csr arrays.
    
    input:
    state_dict: PyTorch state dict for decoder/encoder
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

def sigmoid(z):
    '''
    Computes sigmoid activation.
    
    input:
    z: input vector
    
    output:
    a: output after activation
    '''
    return 1.0/(1.0+np.exp(-z))

def sigmoid_jac(z):
    '''
    Computes sigmoid activation and its derivative.
    
    input:
    z: input vector
    
    output:
    a: output after activation
    '''
    
    emz = np.exp(-z)
    a   = 1.0/(1.0+emz)
    da  = a*a*emz
    return a, da 

def swish(z):
    '''
    Computes swish activation
    
    input:
    z: input vector
    
    output:
    a: output after activation
    '''
    
    return z/(1.0+np.exp(-z))

def swish_jac(z):
    '''
    Computes swish activation and its derivative.
    
    input:
    z: input vector
    
    output:
    a: output after activation
    '''
    
    emz = np.exp(-z)
    
    sig   = 1.0/(1.0+emz)
    dsig  = sig*sig*emz
    
    a   = z*sig
    da  = sig + z*dsig
    return a, da 

class neural_net:
    '''
    Parent class for Decoder and Encoder classes.
    
    fields:
    W1:       hidden layer weight matrix
    b1:       hidden layer bias vector
    W2:       output layer weight matrix
    scale:    scaling vector for normalizing snapshots
    ref:      reference vector for normalizing snapshots
    act_func: activation function
    '''
    def __init__(self, state_dict, scale, ref, mask_shape, act_type):
        '''
        Initialize neural_net class:
        
        inputs: 
        state_dict:  pytorch state_dict containing neural net parameters
        scale:       scaling vector for normalizing snapshots
        ref:         reference vector for normalizing snapshots
        mask_shape:  tuple for dimensions of sparsity mask
        act_type:    activation function type. 'Swish' or 'Sigmoid'
        '''
        self.W1, self.b1, self.W2 = get_net_np_params(state_dict, mask_shape)
        self.scale = scale
        self.ref   = ref
        self.act_func = swish if act_type == 'Swish' else sigmoid
        self.act_type = act_type
        
class Encoder(neural_net):    
    '''
    Encoder class.
    
    fields:
    W1:       hidden layer weight matrix
    b1:       hidden layer bias vector
    W2:       output layer weight matrix
    scale:    scaling vector for normalizing snapshots
    ref:      reference vector for normalizing snapshots
    act_func: activation function
    
    methods:
    fwd:      evaluate encoder
    '''
    def __init__(self, state_dict, scale, ref, mask_shape, act_type):
        super().__init__(state_dict, scale, ref, mask_shape, act_type)
        self.W1 = self.W1@sp_diag(1.0/scale)
        self.b1 = self.b1 - self.W1@self.ref
    def fwd(self, w):
        '''
        Evaluate encoder.
        
        inputs:
        w:        FOM state to encode
        
        output:   encoded FOM state
        '''
        return self.act_func(w@self.W1.T+self.b1)@self.W2.T
#         z1 = (w-self.ref)@self.W1.T+self.b1
#         a1 = self.act_func(z1)
#         return a1@self.W2.T
    
class Decoder(neural_net):
    '''
    Decoder class.
    
    fields:
    W1:       hidden layer weight matrix
    b1:       hidden layer bias vector
    W2:       output layer weight matrix
    scale:    scaling vector for normalizing snapshots
    ref:      reference vector for normalizing snapshots
    act_type: activation function
    
    methods:
    fwd:      evaluate decoder
    fwd_jac:  evaluate decoder and its jacobian
    '''
    def __init__(self, state_dict, scale, ref, mask_shape, act_type):
        super().__init__(state_dict, scale, ref, mask_shape, act_type)
        self.W2 = sp_diag(scale)@self.W2
        self.act_jac  = swish_jac if self.act_type == 'Swish' else sigmoid_jac
        
    def fwd(self, w):
        '''
        Evaluate decoder.
        
        inputs:
        w:        ROM state to decode
        
        output:   decoded FOM state
        '''
        z1 = w@self.W1.T+self.b1
        a1 = self.act_func(z1)
        out = a1@self.W2.T+self.ref
        return out
    
    def fwd_jac(self, w):
        '''
        Evaluate decoder and its jacobian.
        
        inputs:
        w:        ROM state to decode
        
        output:   
        out:      decoder ROM state
        jacobian: jacobian of decoder wrt input w
        '''
        z1 = self.W1@w+self.b1
        a1, da1 = self.act_jac(z1)
        out = self.W2@a1+self.ref
        jac = self.W2@sp_diag(a1)@self.W1
        return out, jac
    
class neural_net_srpc:
    '''
    Parent class for Decoder and Encoder classes for SRPC autoencoders.
    
    fields:
    W1:       hidden layer weight matrix
    b1:       hidden layer bias vector
    W2:       output layer weight matrix
    I:        inclusion matrix for unreduced ROM state components
    scale:    scaling vector for normalizing snapshots
    ref:      reference vector for normalizing snapshots
    act_func: activation function
    '''
    def __init__(self, W1, b1, W2, act_type, I):
        '''
        Initialize neural_net class for interface autoencoder with SRPC:
        
        inputs: 
        W1:       hidden layer weight matrix
        b1:       hidden layer bias vector
        W2:       output layer weight matrix
        act_type: activation function type. 'Swish' or 'Sigmoid'
	I:        inclusion matrix for unreduced ROM state components
        '''
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.I  = I
        self.act_type = act_type
        self.act_func = swish if act_type == 'Swish' else sigmoid

class Encoder_srpc(neural_net_srpc):
    '''
    Encoder class for interface autoencoder with SRPC.
    
    fields:
    W1:       hidden layer weight matrix
    b1:       hidden layer bias vector
    W2:       output layer weight matrix
    act_func: activation function
    I:        inclusion matrix for unreduced ROM state components
    
    methods:
    fwd:      evaluate encoder
    '''
    def fwd(self, w):
        '''
        Evaluate encoder.
        
        inputs:
        w:        FOM state to encode
        
        output:   encoded FOM state
        '''
        return self.act_func(w@self.W1.T+self.b1)@self.W2.T + w@self.I.T

class Decoder_srpc(neural_net_srpc):
    '''
    Decoder class for interface autoencoder with SRPC.
    
    fields:
    W1:       hidden layer weight matrix
    b1:       hidden layer bias vector
    W2:       output layer weight matrix
    ref:      reference vector for normalizing snapshots
    act_func: activation function
    I:        inclusion matrix for unreduced ROM state components
    
    methods:
    fwd:      evaluate decoder
    fwd_jac:  evaluate decoder and its jacobian
    '''
    def __init__(self, W1, b1, W2, act_type, ref, I):
        '''
        Initialize Decoder_srpc class for interface autoencoder
        
        inputs: 
        W1:       hidden layer weight matrix
        b1:       hidden layer bias vector
        W2:       output layer weight matrix
        act_type: activation function type. 'Swish' or 'Sigmoid'
        ref:      reference vector for normalization
	I:        inclusion matrix for unreduced ROM state components
        '''
        super().__init__(W1, b1, W2, act_type, I)
        self.ref = ref
        self.act_jac  = swish_jac if self.act_type == 'Swish' else sigmoid_jac

    def fwd(self, w):
        '''
        Evaluate decoder.
        
        inputs:
        w:        ROM state to decode
        
        output:   decoded FOM state
        '''
        return self.act_func(w@self.W1.T+self.b1)@self.W2.T+self.ref + w@self.I.T
    
    def fwd_jac(self,w):
        '''
        Evaluate decoder and its jacobian.
        
        inputs:
        w:        ROM state to decode
        
        output:   
        out:      decoder ROM state
        jacobian: jacobian of decoder wrt input w
        '''
        z1 = self.W1@w+self.b1
        a1, da1 = self.act_jac(z1)
        out = self.W2@a1+self.ref + self.I@w
        jac = self.W2@sp_diag(a1)@self.W1 + self.I
        return out, jac
    
class nmromhr_state_component:
    '''
    Class for storing hyper reduced quantities for subdomain NM-ROM.
    fields:
    res_nodes:     HR nodes for residual 
    state_nodes:   HR nodes for state component
    cross_nodes:   HR nodes for residual of other state component
    Bx:            hyper reduced Bx matrix
    By:            hyper reduced By matrix
    C:             hyper reduced C matrix
    I:             hyper reduced inclusion matrix I
    Io:            hyper reduced inclusion matrix for other residual component
    '''
    def __init__(self, component, res_nodes, state_nodes, cross_nodes):
        '''
        Instantiate nmromhr_state_component class.
        
        inputs:
        component:     class for FOM state component 
        res_nodes:     HR nodes for residual 
        state_nodes:   HR nodes for state component
        cross_nodes:   HR nodes for state of other state component
        '''
        self.res_nodes = res_nodes
        self.state_nodes = state_nodes
        mat_slice = np.ix_(res_nodes, state_nodes)
        self.Bx = component.Bx[mat_slice]
        self.By = component.By[mat_slice]        
        self.C  = component.C[mat_slice]
        self.I  = component.I[mat_slice]     
        self.Io = component.I[np.ix_(res_nodes, cross_nodes)]
        
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

class subnet:
    '''
    Class for extracting subnet from trained sparse decoder.
    
    fields:
    W1:       hidden layer weight matrix
    b1:       hidden layer bias vector
    W2:       output layer weight matrix
    ref:      reference vector for normalizing snapshots
    I:        inclusion matrix for unreduced ROM state components
    act_type: activation type
    act_jac:  activation function and its derivative
    hr_nodes: indices of HR nodes 
    
    methods:
    fwd:           evaluate decoder
    intr_fwd_jac:  evaluate subnet and its jacobian if AE is for interior states
    intf_sub_fwd:  evaluate subnet and its jacobian if AE for interface states with SRPC
    intf_full_fwd: evaluate decoder and its jacobian, restrict to HR nodes if AE is for interface states with WFPC   
    '''
    def __init__(self, decoder, hr_nodes, comp, srpc=False):
        '''
        Initialize subnet class.
        
        decoder:    decoder from which to extract subnet
        hr_nodes:   HR nodes for subnet evaluation
        comp:       'interior' or 'interface'
        srpc:       Boolean for SRPC formulation
        '''
        ind = get_hr_col_ind(hr_nodes, decoder.W2)
        self.W1 = decoder.W1[ind]
        self.b1 = decoder.b1[ind]
        self.W2 = decoder.W2[np.ix_(hr_nodes, ind)]
        self.ref = decoder.ref[hr_nodes]
        self.act_type = decoder.act_type
        self.act_func = decoder.act_func
        self.act_jac  = decoder.act_jac
        
        self.hr_nodes = hr_nodes
        self.W1_full = decoder.W1
        self.b1_full = decoder.b1
        self.W2_full = decoder.W2
        self.ref_full = decoder.ref
        
        if comp=='interior':
            self.I_full = sp.csr_matrix((decoder.W2.shape[0], decoder.W1.shape[1]))
            self.I      = self.I_full[hr_nodes]
            self.fwd_jac = self.intr_fwd_jac
        else:
            if srpc:
                self.I = decoder.I[hr_nodes]
                self.I_full = decoder.I
                self.fwd_jac = self.intf_sub_fwd
            else:
                self.I_full = sp.csr_matrix((decoder.W2.shape[0], decoder.W1.shape[1]))
                self.I      = self.I_full[hr_nodes]
                self.fwd_jac = self.intf_full_fwd
            
    def fwd(self, w):
        '''
        Evaluate decoder.
        
        inputs:
        w:        ROM state to decode
        
        output:   decoded FOM state
        '''
        return self.act_func(w@self.W1.T+self.b1)@self.W2.T + self.ref + w@self.I.T
        
    def intr_fwd_jac(self,w):
        '''
        Evaluate decoder and its jacobian on subnet.
        
        inputs:
        w:        ROM state to decode
        
        output:   
        out:      decoder ROM state
        jacobian: jacobian of decoder wrt input w
        '''
        z1 = self.W1@w+self.b1
        a1, da1 = self.act_jac(z1)
        out = self.W2@a1+self.ref 
        jac = self.W2@sp_diag(a1)@self.W1 
        return out, jac
    
    def intf_sub_fwd(self,w):
        '''
        Evaluate decoder and its jacobian on subnet.
        
        inputs:
        w:        ROM state to decode
        
        output:   
        out:      decoder ROM state
        jacobian: jacobian of decoder wrt input w
        '''
        z1 = self.W1@w+self.b1
        a1, da1 = self.act_jac(z1)
        out = self.W2@a1+self.ref + self.I@w
        jac = self.W2@sp_diag(a1)@self.W1 + self.I
        return out, jac, out, jac
    
    def intf_full_fwd(self, w):
        '''
        Evaluate decoder and its jacobian on full network, then restrict to HR nodes.
        
        inputs:
        w:        ROM state to decode
        
        output:   
        out:      decoder ROM state
        jacobian: jacobian of decoder wrt input w
        '''
        z1 = self.W1_full@w+self.b1_full
        a1, da1 = self.act_jac(z1)
        out = self.W2_full@a1+self.ref_full #+ self.I_full@w
        jac = self.W2_full@sp_diag(a1)@self.W1_full #+ self.I_full
        return out, jac, out[self.hr_nodes], jac[self.hr_nodes]
    
class burgers_nmrom_component:
    '''
    Generate ROM state component for 2D Burgers' equation from basis
    
    fields: 
    scale:      scaling vector for normalizing snapshots
    ref:        reference vector for normalizing snapshots
    mask:       sparsity mask for decoder output layer
    fomsize:    dimension of FOM state component
    romsize:    dimension of ROM state component
    act_type:   activation type. 'Swish' or 'Sigmoid'
    train_time: time to train autoencoder 
    encoder:    instance of Encoder class for current state component
    decoder:    instance of Decoder class for current state component
    
    Bx:         Bx matrix for Burgers' FOM
    BY:         By matrix for Burgers' FOM
    C:          C matrix for Burgers' FOM
    I:          inclusion matrix for Burgers' FOM
    XY:         FD nodes in current subdomain component
    indices:    indices of monolithic FOM state corresponding to current state component 
    
    methods:
    set_initial: set initial condition for current state
    '''
    def __init__(self, component, ae_dict, comp='interior',
                 res_size=-1, hr_nodes=[],
                 in_ports=[], rom_port_ind={}, fom_port_ind={}, srpc=False):
        
        self.indices = component.indices
        self.fomsize = self.indices.size
        self.XY = component.XY
        
        # assemble autoencoder for SRPC case
        if srpc:
            assert len(ae_dict)==len(in_ports)
            de_W1, de_b1, de_W2, en_W1, en_b1, en_W2, I, self.ref, self.mask, act_type, self.romsize, self.train_time =\
                        assemble_srpc_ae_params(ae_dict, in_ports, fom_port_ind, rom_port_ind, 2*self.fomsize)
            
            self.encoder = Encoder_srpc(en_W1, en_b1, en_W2, act_type, I.T)
            self.decoder = Decoder_srpc(de_W1, de_b1, de_W2, act_type, self.ref, I)
        
        # assembler autoencoder for WPFC case
        else:
            self.scale = ae_dict['scale'].to('cpu').detach().numpy()
            self.ref   = ae_dict['ref'].to('cpu').detach().numpy()
            self.mask  = ae_dict['mask']
            self.romsize  = ae_dict['latent_dim']
            act_type        = ae_dict['act_type']
            self.train_time = ae_dict['train_time']

            self.encoder = Encoder(ae_dict['encoder'], 
                                   self.scale, 
                                   self.ref, 
                                   self.mask.T.shape, 
                                   act_type)
            self.decoder = Decoder(ae_dict['decoder'], 
                                   self.scale, 
                                   self.ref, 
                                   self.mask.shape, 
                                   act_type)
        
        # compute subnets if HR is applied
        if len(hr_nodes)>0:
#             self.hr_nodes = hr_nodes
            hr_u_res = hr_nodes[hr_nodes < res_size]
            hr_v_res = hr_nodes[hr_nodes >= res_size]-res_size
            
            hr_uv_union = list(set(hr_u_res).union(set(hr_v_res)))
            
            hr_u_state = get_hr_col_ind(hr_uv_union, [component.Bx, component.By, component.C, component.I])
            hr_v_state = get_hr_col_ind(hr_uv_union, [component.Bx, component.By, component.C, component.I])
            
            self.u = nmromhr_state_component(component, hr_u_res, hr_u_state, hr_v_state)
            self.v = nmromhr_state_component(component, hr_v_res, hr_v_state, hr_u_state)
            
            self.subdecoder = subnet(self.decoder, 
                                     np.concatenate([hr_u_state, hr_v_state+self.fomsize]).astype(int),
                                     comp, srpc=srpc)
        else:
            self.Bx = component.Bx
            self.By = component.By
            self.C  = component.C
            self.I  = component.I
        
    def set_initial(self, u0, v0):
        '''
        Set initial condition.
        
        inputs:
        u0:  function for initial u
        v0:  function for initial v
        '''
        self.w0 = self.encoder.fwd(np.concatenate([u0(self.XY), v0(self.XY)]))
        
# assemble interface encoder and decoder
def assemble_srpc_ae_params(port_dict, in_ports, fom_port_ind, rom_port_ind, fomsize):
    '''
    Assemble weights and biases for SRPC-based autoencoders.
    
    inputs:
    port_dict:     dictionary of dictionaries.
                   port_dict[p] = autoencoder dictionary corresponding to port p
    in_ports:      list of ports that current subdomain belongs to
    fom_port_ind:  list of arrays where
                   fom_port_ind[p] = indices of FOM intf state on subdomain j corresponding to port p
    rom_port_ind:  list of arrays where
                   rom_port_ind[p] = indices of ROM intf state on subdomain j corresponding to port p
    fomsize:       dimension of FOM interface state on current subdomain
    
    outputs:
    de_W1:           W1 weight matrix for decoder
    de_b1:           b1 bias vector for decoder
    de_W2:           W2 weight matrix for decoder
    en_W1:           W1 weight matrix for encoder
    en_b1:           b1 bias vector for encoder
    en_W2:           W2 weight matrix for encoder
    I:               inclusion matrix for unreduced ROM state components
    ref:             reference vector 
    mask:            sparsity mask for decoder output later
    act_type:        activation type. 'Swish' or 'Sigmoid'
    romsize:         dimension of ROM interface state
    train_time_port: dictionary where train_time_port[p] = train time for port autoencoder p
    '''
    de_W1_list = []
    de_b1_list = []
    de_W2_list = []
    en_W1_list = []
    en_b1_list = []
    en_W2_list = []
    romsize = np.sum([min(port_dict[p]['latent_dim'], len(fom_port_ind[p])) for p in port_dict])
    Icol = []
    Irow = []
    Idata = []
    ref = np.zeros(fomsize)
    train_time_port = {}
    act_type = port_dict[in_ports[0]]['act_type']
    for j in in_ports:
        
        if len(rom_port_ind[j])<len(fom_port_ind[j]):
            train_time_port[j]=port_dict[j]['train_time']

            scale = port_dict[j]['scale'].to('cpu').detach().numpy()
            ref_i = port_dict[j]['ref'].to('cpu').detach().numpy()

            # decoder
            mask       = port_dict[j]['mask']
            W1, b1, W2 = get_net_np_params(port_dict[j]['decoder'], mask.shape)
            W1_block   = np.zeros((port_dict[j]['decoder_hidden'], romsize))
            W1_block[:, rom_port_ind[j]] += W1
            de_W1_list.append(sp.csr_matrix(W1_block))
            de_b1_list.append(b1)

            W2       = sp_diag(scale)@W2
            W2_block = np.zeros((fomsize, port_dict[j]['decoder_hidden']))
            W2_block[fom_port_ind[j], :] += W2
            de_W2_list.append(sp.csr_matrix(W2_block))

            # encoder
            W1, b1, W2 = get_net_np_params(port_dict[j]['encoder'], mask.T.shape)
            W1 = W1@sp_diag(1.0/scale)
            W1_block = np.zeros((port_dict[j]['encoder_hidden'], fomsize))
            W1_block[:, fom_port_ind[j]] += W1
            en_W1_list.append(sp.csr_matrix(W1_block))

            en_b1_list.append(b1-W1@ref_i)

            W2_block = np.zeros((romsize, port_dict[j]['encoder_hidden']))
            W2_block[rom_port_ind[j], :] += W2
            en_W2_list.append(sp.csr_matrix(W2_block))

            ref[fom_port_ind[j]] += ref_i
        else: 
            Icol.append(rom_port_ind[j])
            Irow.append(fom_port_ind[j])
            Idata.append(np.ones(len(fom_port_ind[j])))
            
    de_W1 = sp.vstack(de_W1_list)
    de_W2 = sp.hstack(de_W2_list)
    de_b1 = np.concatenate(de_b1_list)
    en_W1 = sp.vstack(en_W1_list)
    en_W2 = sp.hstack(en_W2_list)
    en_b1 = np.concatenate(en_b1_list)
    if len(Idata)>0:
        I = sp.coo_matrix((np.concatenate(Idata), (np.concatenate(Irow), np.concatenate(Icol))), shape=(fomsize, romsize)).tocsr()
    else: 
        I = sp.csr_matrix((fomsize, romsize))
    mask = copy(de_W2)
    mask.data = np.ones_like(mask.data)
    
    return de_W1, de_b1, de_W2, en_W1, en_b1, en_W2, I, ref, mask, act_type, romsize, train_time_port

class subdomain_nmrom:
    '''
    Generate NM-ROM subdomain class for DD formulation of 2D Burgers' equation.
    
    fields:
    constraint_mat:    compatibility constraint matrix
    in_ports:          list of ports that current subdomain belongs to
    residual_ind:      indices of FOM corresponding to residual states in current subdomain
    interior:          instance of burgers_nmrom_component class corresponding to interior states
    interface:         instance of burgers_nmrom_component class corresponding to interface states
    spzero:            sparse zero matrix for constructing KKT system
    
    methods:
    set_initial:       set initial condition for current subdomain
    res_jac:           compute residual and residual jacobian
    '''
    def __init__(self, subdomain, constraint_mat, intr_dict, intf_dict={}, port_dict={},
                 rom_port_ind={}, fom_port_ind={},
                 constraint_type='wfpc', scaling=1):
        '''
        initialize subdomain nm-rom class.
        
        inputs:
        subdomain:       instance of subdomain_fom class corresponding to current subdomain
        constraint_mat:  constraint matrix corresponding to current subdomain
        intr_dict:       dictionary for interior autoencoder
        intf_dict:       dictionary for interface autoencoder
        port_port:       dictionary of dictionarys for port autoencoders comprising interface autoencoder
        fom_port_ind:    list of arrays where
                         fom_port_ind[p] = indices of FOM intf state on subdomain j corresponding to port p
        rom_port_ind:    list of arrays where
                         rom_port_ind[p] = indices of ROM intf state on subdomain j corresponding to port p
        constraint_type: 'wfpc' or 'srpc'
        '''
        self.in_ports = subdomain.in_ports
        
        self.residual_ind   = subdomain.residual_ind
        self.constraint_mat = constraint_mat
        
        self.scaling  = scaling
        self.interior = burgers_nmrom_component(subdomain.interior, intr_dict)
        
        self.spzero = sp.csr_matrix((self.constraint_mat.shape[0], self.interior.romsize))
        
        if constraint_type == 'wfpc':
            assert len(intf_dict) >0
            self.hstack = np.hstack
            self.constraint_eval = lambda xhat, g, dg: (self.constraint_mat@g, self.constraint_mat@dg)
            self.interface = burgers_nmrom_component(subdomain.interface, intf_dict, srpc=False)
            
        else:
            assert len(port_dict) >0
            assert len(rom_port_ind)>0
            assert len(fom_port_ind)>0
            self.hstack = sp.hstack
            
            self.constraint_eval = lambda xhat, g, dg: (self.constraint_mat@xhat, self.constraint_mat)
            self.interface = burgers_nmrom_component(subdomain.interface, 
                                                     port_dict,
                                                     in_ports=self.in_ports,
                                                     rom_port_ind=rom_port_ind,
                                                     fom_port_ind=fom_port_ind,
                                                     srpc=True)
            
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
        Ag:  constraint matrix times interface state
        Adg: constraint matrix times derivative of interface state wrt rom state
        '''
        uvn_intr, de_intr_jac = self.interior.decoder.fwd_jac(wn_intr)
        un_intr  = uvn_intr[:self.interior.fomsize]
        vn_intr  = uvn_intr[self.interior.fomsize:]
        
        uvn_intf, de_intf_jac = self.interface.decoder.fwd_jac(wn_intf)
        un_intf  = uvn_intf[:self.interface.fomsize]
        vn_intf  = uvn_intf[self.interface.fomsize:]    
        
        uvc_intr = self.interior.decoder.fwd(wc_intr)
        uc_intr  = uvc_intr[:self.interior.fomsize]
        vc_intr  = uvc_intr[self.interior.fomsize:]
        
        uvc_intf = self.interface.decoder.fwd(wc_intf)
        uc_intf  = uvc_intf[:self.interface.fomsize]
        vc_intf  = uvc_intf[self.interface.fomsize:]  
        
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
        jac_intr = sp.bmat([[Juu_intr, Juv_intr], [Jvu_intr, Jvv_intr]], format='csr')@de_intr_jac
        
        # compute interface state jacobian
        UBx_VBy_C_intf = sp_diag(un_res)@self.interface.Bx + sp_diag(vn_res)@self.interface.By + self.interface.C
        Juu_intf = self.interface.I - ht*(sp_diag(Bxu)@self.interface.I + UBx_VBy_C_intf)
        Juv_intf = -ht*sp_diag(Byu)@self.interface.I
        Jvu_intf = -ht*sp_diag(Bxv)@self.interface.I
        Jvv_intf = self.interface.I - ht*(sp_diag(Byv)@self.interface.I + UBx_VBy_C_intf)
        jac_intf = sp.bmat([[Juu_intf, Juv_intf], [Jvu_intf, Jvv_intf]], format='csr')@de_intf_jac
        
        # compute terms needed for SQP solver
#         Ag  = self.constraint_mat@uvn_intf
        Ag, Adg = self.constraint_eval(wn_intf, uvn_intf, de_intf_jac)
        
        jac = self.hstack([jac_intr, jac_intf])
        H   = self.scaling*(jac.T@jac)
        rhs = np.concatenate([self.scaling*(jac_intr.T@res), 
                              self.scaling*(jac_intf.T@res) + Adg.T@lam])
        
        return res, jac, H, rhs, Ag, Adg
    
class subdomain_nmrom_hr:
    '''
    Generate NM-ROM-HR subdomain class for DD formulation of 2D Burgers' equation.
    
    fields:
    constraint_mat:    compatibility constraint matrix
    in_ports:          list of ports that current subdomain belongs to
    residual_ind:      indices of FOM corresponding to residual states in current subdomain
    interior:          instance of burgers_nmrom_component class corresponding to interior states
    interface:         instance of burgers_nmrom_component class corresponding to interface states
    spzero:            sparse zero matrix for constructing KKT system
    
    methods:
    set_initial:       set initial condition for current subdomain
    res_jac:           compute residual and residual jacobian
    '''
    def __init__(self, subdomain, constraint_mat, res_basis, intr_dict, intf_dict={}, 
                 port_dict={}, rom_port_ind={}, fom_port_ind={},
                 constraint_type='wfpc', hr_type='collocation',
                 nz=-1, ncol=-1, n_corners=5, scaling=1):
        '''
        initialize subdomain nm-rom class.
        
        inputs:
        subdomain:       instance of subdomain_fom class corresponding to current subdomain
        constraint_mat:  constraint matrix corresponding to current subdomain
        res_basis:       basis for residual to be used in computation of HR nodes
        intr_dict:       dictionary for interior autoencoder
        intf_dict:       dictionary for interface autoencoder
        port_port:       dictionary of dictionarys for port autoencoders comprising interface autoencoder
        fom_port_ind:    list of arrays where
                         fom_port_ind[p] = indices of FOM intf state on subdomain j corresponding to port p
        rom_port_ind:    list of arrays where
                         rom_port_ind[p] = indices of ROM intf state on subdomain j corresponding to port p
        constraint_type: 'wfpc' or 'srpc'
        hr_type:         'collocation' or 'gappy_POD'
        nz:              number of hyper reduction indices
        ncol:            number of working columns for sample node selection algorithm
        n_corners:       [optional] number of interface columns to include in sample nodes. Default is 5
        '''
        self.in_ports       = subdomain.in_ports
        self.residual_ind   = subdomain.residual_ind
        self.constraint_mat = constraint_mat
        self.scaling        = scaling
        
        self.res_size = res_basis.shape[0]//2
        self.hr_nodes = select_sample_nodes(subdomain, res_basis, self.res_size, nz, ncol, n_corners=n_corners)
        
        if hr_type == 'gappy_POD':
            self.gappy_basis = np.linalg.pinv(res_basis[self.hr_nodes])
            self.hr_func     = lambda res, jac_intr, jac_intf: (self.gappy_basis@res, 
                                                                self.gappy_basis@jac_intr,
                                                                self.gappy_basis@jac_intf)
        else:
            self.hr_func     = lambda res, jac_intr, jac_intf: (res, jac_intr, jac_intf)
            
        self.interior = burgers_nmrom_component(subdomain.interior, intr_dict, comp='interior',
                                                res_size=self.res_size, hr_nodes=self.hr_nodes, srpc=False)
        self.spzero   = sp.csr_matrix((self.constraint_mat.shape[0], self.interior.romsize))
        
        if constraint_type == 'wfpc':
            assert len(intf_dict) >0
            self.hstack = np.hstack
            self.constraint_eval = lambda xhat, g, dg: (self.constraint_mat@g, self.constraint_mat@dg)
            self.interface = burgers_nmrom_component(subdomain.interface, intf_dict, comp='interface', srpc=False,
                                                     res_size=self.res_size, hr_nodes=self.hr_nodes)
            
        else:
            assert len(port_dict) >0
            assert len(rom_port_ind)>0
            assert len(fom_port_ind)>0
            self.hstack = sp.hstack if hr_type == 'collocation' else np.hstack
            
            self.constraint_eval = lambda xhat, g, dg: (self.constraint_mat@xhat, self.constraint_mat)
            self.interface = burgers_nmrom_component(subdomain.interface, 
                                                     port_dict,
                                                     comp='interface',
                                                     in_ports=self.in_ports,
                                                     rom_port_ind=rom_port_ind,
                                                     fom_port_ind=fom_port_ind,
                                                     srpc=True, 
                                                     res_size=self.res_size, hr_nodes=self.hr_nodes)
            
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
        Ag:  constraint matrix times interface state
        Adg: constraint matrix times derivative of interface state wrt rom state
        '''
        uvn_intr, de_intr_jac = self.interior.subdecoder.fwd_jac(wn_intr)
        un_intr  = uvn_intr[:self.interior.u.state_nodes.size]
        vn_intr  = uvn_intr[self.interior.u.state_nodes.size:]
        
        uvn_intf, de_intf_jac, uvn_intf_hr, de_intf_jac_hr = self.interface.subdecoder.fwd_jac(wn_intf)
        un_intf  = uvn_intf_hr[:self.interface.u.state_nodes.size]
        vn_intf  = uvn_intf_hr[self.interface.u.state_nodes.size:]    
        
        uvc_intr = self.interior.subdecoder.fwd(wc_intr)
        uc_intr  = uvc_intr[:self.interior.u.state_nodes.size]
        vc_intr  = uvc_intr[self.interior.u.state_nodes.size:]
        
        uvc_intf = self.interface.subdecoder.fwd(wc_intf)
        uc_intf  = uvc_intf[:self.interface.u.state_nodes.size]
        vc_intf  = uvc_intf[self.interface.u.state_nodes.size:]  
                
        # store relevant quantities
        uun_res = self.interior.u.I@un_intr + self.interface.u.I@un_intf
        uvn_res = self.interior.u.Io@vn_intr + self.interface.u.Io@vn_intf
        vun_res = self.interior.v.Io@un_intr + self.interface.v.Io@un_intf
        vvn_res = self.interior.v.I@vn_intr + self.interface.v.I@vn_intf
        
        uuc_res = self.interior.u.I@uc_intr + self.interface.u.I@uc_intf
        vvc_res = self.interior.v.I@vc_intr + self.interface.v.I@vc_intf
        
        Bxu = self.interior.u.Bx@un_intr + self.interface.u.Bx@un_intf
        Byu = self.interior.u.By@un_intr + self.interface.u.By@un_intf
        Cu  = self.interior.u.C@un_intr + self.interface.u.C@un_intf
        
        Bxv = self.interior.v.Bx@vn_intr + self.interface.v.Bx@vn_intf
        Byv = self.interior.v.By@vn_intr + self.interface.v.By@vn_intf
        Cv  = self.interior.v.C@vn_intr + self.interface.v.C@vn_intf
        
        # compute residuals
        res_u = uun_res - uuc_res - ht*(uun_res*Bxu + uvn_res*Byu + Cu)
        res_v = vvn_res - vvc_res - ht*(vun_res*Bxv + vvn_res*Byv + Cv)
        res   = np.concatenate([res_u, res_v])
        
        # compute interior state jacobian
        Juu_intr = self.interior.u.I - ht*(sp_diag(Bxu)@self.interior.u.I + sp_diag(uun_res)@self.interior.u.Bx+
                                           sp_diag(uvn_res)@self.interior.u.By + self.interior.u.C)
        Juv_intr = -ht*sp_diag(Byu)@self.interior.u.Io
        Jvu_intr = -ht*sp_diag(Bxv)@self.interior.v.Io
        Jvv_intr = self.interior.v.I - ht*(sp_diag(Byv)@self.interior.v.I + sp_diag(vvn_res)@self.interior.v.By+
                                          sp_diag(vun_res)@self.interior.v.Bx + self.interior.v.C)
        jac_intr = sp.bmat([[Juu_intr, Juv_intr], [Jvu_intr, Jvv_intr]], format='csr')@de_intr_jac
        
        # compute interface state jacobian
        Juu_intf = self.interface.u.I - ht*(sp_diag(Bxu)@self.interface.u.I + sp_diag(uun_res)@self.interface.u.Bx+
                                           sp_diag(uvn_res)@self.interface.u.By + self.interface.u.C)
        Juv_intf = -ht*sp_diag(Byu)@self.interface.u.Io
        Jvu_intf = -ht*sp_diag(Bxv)@self.interface.v.Io
        Jvv_intf = self.interface.v.I - ht*(sp_diag(Byv)@self.interface.v.I + sp_diag(vvn_res)@self.interface.v.By+
                                          sp_diag(vun_res)@self.interface.v.Bx + self.interface.v.C)
        jac_intf = sp.bmat([[Juu_intf, Juv_intf], [Jvu_intf, Jvv_intf]], format='csr')@de_intf_jac_hr
        res, jac_intr, jac_intf = self.hr_func(res, jac_intr, jac_intf)
        
        # compute terms needed for SQP solver
#         Ag  = self.constraint_mat@uvn_intf
        Ag, Adg = self.constraint_eval(wn_intf, uvn_intf, de_intf_jac)
        
        jac = self.hstack([jac_intr, jac_intf])
        H   = self.scaling*(jac.T@jac)
        rhs = np.concatenate([self.scaling*(jac_intr.T@res),
                              self.scaling*(jac_intf.T@res) + Adg.T@lam])
        
        return res, jac, H, rhs, Ag, Adg  
    
class DD_NMROM:
    '''
    Class for DD NM-ROM applied to the 2D time-dependent Burgers' equation.
    
    fields:
    nxy:       size of monolithic FOM
    n_sub:     number of subdomains
    ports:     ports for DD configuration.
               ports[j] = frozenset containing indices of subdomains in port j
    subdomain: list of instances of subdomain NM-ROM classes
    '''
    def __init__(self, ddfom, 
                 intr_ae_list,
                 intf_ae_list=[],
                 port_ae_list=[],
                 res_bases=[], 
                 constraint_type='srpc', 
                 hr=True,
                 hr_type='collocation',
                 sample_ratio=2,
                 n_samples=-1,
                 n_corners=-1,
                 n_constraints=1, 
                 seed=None, 
                 scaling=1):
        
        '''
        Instantiate DD NM-ROM class.
        
        inputs:
        intr_ae_list:    list of dictionaries containing interior state autoencoder information for each subdomain
        intf_ae_list:    list of dictionaries containing interface state autoencoder information for each subdomain
        port_ae_list:    list of dictionaries containing port state autoencoder information for each port
        res_bases:       list of residual bases used for HR
        constraint_type: [optional] Constraint formulation. 'wfpc' or 'srpc'. Default is 'srpc'
        hr:              [optional] Boolean for enabling HR. Default is True
        hr_type:         [optional] 'collocation' or 'gappy_POD'. Default is 'collocation'
        sample_ratio:    [optional] ratio of number of hyper-reduction samples to residual basis size. Default is 2
        n_samples:       [optional] specify number of hyper reduction sample nodes. 
                           If n_samples is an array with length equal to the number of subdomains, then 
                           n_samples[i] is the number of HR samples on the ith subdomain.
                
                           If n_samples is a positive integer, then each subdomain has n_samples HR nodes. 
                
                           Otherwise, the number of samples is determined by the sample ratio. 
                           Default is -1. 
                
        n_corners:       [optional] Number of interface nodes included in the HR sample nodes. 
                            If n_corners is an array with length equal to the number of subdomains, then 
                            n_corners[i] is the number of interface HR nodes on the ith subdomain.
                 
                            If n_corners is a positive integer, then each subdomain has n_corners interface HR nodes. 
                
                            Otherwise, the number of interface HR nodes on each subdomain is determined by n_samples
                            multiplied by the ratio of the number of interface nodes contained in the residual nodes 
                            to the total number of residual nodes.
                
                            Default is -1. 
        seed:            [optional] random seed. Default is None.
        '''
        self.nxy       = ddfom.nxy
        self.hxy       = ddfom.hxy
        self.n_sub     = ddfom.n_sub
        self.ports     = ddfom.ports
        self.hr        = hr
        self.subdomain = []
        self.scaling   = self.hxy if scaling < 0 else scaling
        
        if constraint_type=='srpc':
            assert len(port_ae_list) == len(ddfom.ports)
            
            # assign coupling conditions to ROM states
            # rom_port_ind[j][p] = indices of ROM intf state on subdomain j corresponding to port p
            # fom_port_ind[j][p] = indices of FOM intf state on subdomain j corresponding to port p
            self.rom_port_ind = []
            self.fom_port_ind = []
            n_intf_list = []
            for s in ddfom.subdomain:
                rom_port_dict = {}
                fom_port_dict = {}
                shift = 0
                for p in s.in_ports:
                    p_ind = np.concatenate([ddfom.port_dict[ddfom.ports[p]], ddfom.port_dict[ddfom.ports[p]]+ddfom.nxy])
                    s_ind = np.concatenate([s.interface.indices, s.interface.indices+ddfom.nxy])
                    fom_port_dict[p] = np.nonzero(np.isin(s_ind, p_ind))[0]
                    sz = min(port_ae_list[p]['latent_dim'], len(fom_port_dict[p]))
                    rom_port_dict[p] = np.arange(sz)+shift
                    shift += sz
                self.rom_port_ind.append(rom_port_dict)
                self.fom_port_ind.append(fom_port_dict)
                n_intf_list.append(shift)

            # assemble ROM-port constraint matrices
            self.n_constraints  = np.sum([(len(port)-1)*port_ae_list[k]['latent_dim'] for k, port in enumerate(ddfom.ports)])
            constraint_mat_list = [sp.coo_matrix((self.n_constraints, n_intf)) for n_intf in n_intf_list]
            
            shift = 0 
            for j, p in enumerate(ddfom.ports):
                port = list(p)
                npj = port_ae_list[j]['latent_dim']
                for i in range(len(port)-1):
                    s1   = port[i]
                    constraint_mat_list[s1].col  = np.concatenate((constraint_mat_list[s1].col, self.rom_port_ind[s1][j]))
                    constraint_mat_list[s1].row  = np.concatenate((constraint_mat_list[s1].row, np.arange(npj)+shift))
                    constraint_mat_list[s1].data = np.concatenate((constraint_mat_list[s1].data, np.ones(npj)))   

                    s2   = port[i+1]
                    constraint_mat_list[s2].col  = np.concatenate((constraint_mat_list[s2].col, self.rom_port_ind[s2][j]))
                    constraint_mat_list[s2].row  = np.concatenate((constraint_mat_list[s2].row, np.arange(npj)+shift))
                    constraint_mat_list[s2].data = np.concatenate((constraint_mat_list[s2].data, -np.ones(npj)))

                    shift += npj
                    
        else:
            assert len(intf_ae_list)==self.n_sub
            
            self.n_constraints = n_constraints
            rng = np.random.default_rng(seed)
            self.constraint_mult = rng.standard_normal(size=(self.n_constraints, ddfom.n_constraints))
            constraint_mat_list  = [self.constraint_mult@s.constraint_mat for j, s in enumerate(ddfom.subdomain)]
                
        if hr:
            self.hr_type   = hr_type
            
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
            n_corners_max = nz_max - np.array([2*s.interior.size for s in ddfom.subdomain])
            if isinstance(n_corners, int):
                if n_corners > 0:
                    n_corners = n_corners*np.ones(self.n_sub)
                else:
                    n_corners = nz*3//4 #np.round(nz*n_corners_max/nz_max)
            n_corners = np.minimum(n_corners, n_corners_max)
            
            # instantiate subdomain NM-ROMs
            if constraint_type=='srpc':    # srpc formulation
                for i, s in enumerate(ddfom.subdomain):
                    port_dict = {}
                    for j in s.in_ports: port_dict[j] = port_ae_list[j]
                    self.subdomain.append(subdomain_nmrom_hr(s,
                                                             constraint_mat_list[i], 
                                                             res_bases[i],
                                                             intr_ae_list[i],
                                                             port_dict=port_dict,
                                                             rom_port_ind=self.rom_port_ind[i],
                                                             fom_port_ind=self.fom_port_ind[i], 
                                                             constraint_type=constraint_type, 
                                                             hr_type=self.hr_type,
                                                             nz=int(nz[i]),
                                                             ncol=int(ncol[i]),
                                                             n_corners=int(n_corners[i]),
                                                             scaling=self.scaling))
            else:      # wfpc formulation
                for i, s in enumerate(ddfom.subdomain):
                    self.subdomain.append(subdomain_nmrom_hr(s, 
                                                             constraint_mat_list[i], 
                                                             res_bases[i],
                                                             intr_ae_list[i],
                                                             intf_dict=intf_ae_list[i],
                                                             constraint_type=constraint_type,
                                                             hr_type=self.hr_type,
                                                             nz=int(nz[i]),
                                                             ncol=int(ncol[i]),
                                                             n_corners=int(n_corners[i]), 
                                                             scaling=self.scaling))
            
        else:    # no HR
            if constraint_type=='srpc':     # SRPC formulation
                for i, s in enumerate(ddfom.subdomain):
                    port_dict = {}
                    for j in s.in_ports: port_dict[j] = port_ae_list[j]
                    self.subdomain.append(subdomain_nmrom(s,
                                                         constraint_mat_list[i], 
                                                         intr_ae_list[i],
                                                         port_dict=port_dict,
                                                         rom_port_ind=self.rom_port_ind[i],
                                                         fom_port_ind=self.fom_port_ind[i], 
                                                         constraint_type=constraint_type, 
                                                         scaling=self.scaling))
            else:      # WFPC formulation
                for i, s in enumerate(ddfom.subdomain):
                    self.subdomain.append(subdomain_nmrom(s, 
                                                          constraint_mat_list[i], 
                                                          intr_ae_list[i],
                                                          intf_dict=intf_ae_list[i],
                                                          constraint_type=constraint_type, 
                                                          scaling=self.scaling))
                
    
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
            interior_ind  = np.arange(s.interior.romsize)
            interface_ind = np.arange(s.interface.romsize)

            wn_intr = wn[interior_ind+shift]
            wc_intr = wc[interior_ind+shift]
            shift  += s.interior.romsize

            wn_intf = wn[interface_ind+shift]
            wc_intf = wc[interface_ind+shift]
            shift  += s.interface.romsize
            
            # computes residual, jacobian, and other quantities needed for KKT system
            res, jac, H, rhs_i, Ag, Adg = s.res_jac(wn_intr, wn_intf, 
                                                    wc_intr, wc_intf, lam, ht)
            stimes[i] = time()-start
            
            # RHS block for KKT system
            start = time()
            rhs.append(rhs_i)
            constraint_res += Ag
            A_list += [s.spzero, Adg] 
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
    
    def solve(self, t_lim, nt, 
              guess=[], 
              tol=1e-9, maxit=20, print_hist=False):
        '''
        Solve DD NM-ROM for 2D Burgers' IVP. 
        
        inputs:
        t_lim:      t_lim[0] = initial time
                    t_lim[1] = final time 
        nt:         number of time steps
        guess:      (nt,nD+nA) array for SQP solver initial guess at each time step 
        tol:        [optional] solver relative tolerance. Default is 1e-10
        maxit:      [optional] maximum number of iterations. Default is 20
        print_hist: [optional] Set to True to print iteration history. Default is False
        
        outputs:
        uu:         list. uu[i] = array for u component of Burgers' equation solution on ith subdomain
        vv:         list. vv[i] = array for v component of Burgers' equation solution on ith subdomain
        runtime:    wall clock time for solve
        '''
        use_guess = True if len(guess)>0 else False 
        
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
                                                                guess[k] if use_guess else wc,
                                                                tol=tol, 
                                                                maxit=maxit, 
                                                                print_hist=print_hist)
            runtime += rtk
            iter_hist.append(iter)

            start=time()
            wc = y
            ww.append(y)
#             if flag == 1: 
#                 print(f'Time step {k+1}: solver failed to converge in {maxit} iterations.')
#                 print(f'                 terminal gradient norm = {norm_res[-1]:1.4e}')
#                 sys.stdout.flush()
#                 break
            if flag == 2:
                print(f'Time step {k+1}: no stepsize found at iteration {iter}.')
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
            w_intr_i = ww[:, shift:s.interior.romsize+shift]
            shift += s.interior.romsize
            w_intf_i = ww[:, shift:s.interface.romsize+shift]
            shift += s.interface.romsize
            
            w_intr.append(w_intr_i)
            w_intf.append(w_intf_i)
            
            uv_intr_i = s.interior.decoder.fwd(w_intr_i)
            u_intr_i, v_intr_i = uv_intr_i[:, :s.interior.fomsize], uv_intr_i[:, s.interior.fomsize:]
            uv_intf_i = s.interface.decoder.fwd(w_intf_i)
            u_intf_i, v_intf_i = uv_intf_i[:, :s.interface.fomsize], uv_intf_i[:, s.interface.fomsize:]
            
#             u_intr_i = np.zeros((ww.shape[0], s.interior.fomsize))
#             v_intr_i = np.zeros((ww.shape[0], s.interior.fomsize))
#             u_intf_i = np.zeros((ww.shape[0], s.interface.fomsize))
#             v_intf_i = np.zeros((ww.shape[0], s.interface.fomsize))
            
#             for k in range(ww.shape[0]):
#                 uv_intr_i   = s.interior.decoder.fwd(w_intr_i[k])
#                 u_intr_i[k] = uv_intr_i[:s.interior.fomsize]
#                 v_intr_i[k] = uv_intr_i[s.interior.fomsize:]
                
#                 uv_intf_i   = s.interface.decoder.fwd(w_intf_i[k])
#                 u_intf_i[k] = uv_intf_i[:s.interface.fomsize]
#                 v_intf_i[k] = uv_intf_i[s.interface.fomsize:]
                
            u_intr.append(u_intr_i)
            v_intr.append(v_intr_i)
            u_intf.append(u_intf_i)
            v_intf.append(v_intf_i)
            
            u_full[:, s.interior.indices] = u_intr_i
            v_full[:, s.interior.indices] = v_intr_i
            u_full[:, s.interface.indices] = u_intf_i
            v_full[:, s.interface.indices] = v_intf_i
            
        return w_intr, w_intf, u_intr, v_intr, u_intf, v_intf, u_full, v_full, lam, runtime, iter_hist, flag
