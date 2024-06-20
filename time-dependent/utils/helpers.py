import numpy as np
import scipy.sparse as sp

def sp_diag(vec):
    '''
    Helper function to compute sparse diagonal matrix.

    input:
    vec: vector to compute sparse diagonal matrix of

    output:
    D: sparse diagonal matrix with diagonal vec
    '''
    return sp.spdiags(vec, 0, vec.size, vec.size)

# select sample nodes for hyper-reduction
def select_sample_nodes(subdomain, residual_basis, res_size, nz, ncol, n_corners=1):
    '''
    Greedy algorithm to select sample nodes for hyper reduction.
    inputs:
        subdomain:       instance of class subdomain_model
        residual_basis:  array of residual basis vectors corresponding to subdomain
        res_size:        dimension of residual subdomain
        nz:              integer of desired number of sample nodes
        ncol:            number of working columns of residual_basis
        n_corners:       number of interface nodes to include in sample nodes
        
    outputs:
        s: array of sample nodes
    '''
    s = np.array([], dtype=np.int)
    
    # greedily sample corner nodes
    if n_corners > 0:
        corners = subdomain.interface.I.tocoo().row
        corners = np.concatenate([corners, corners+res_size])
        corner_ind = select_sample_nodes(subdomain, residual_basis[corners], res_size, n_corners, ncol, n_corners=-1)
        s = np.union1d(s, corners[corner_ind])
        
    na       = nz - s.size                             # number of additional nodes to sample
    nb       = 0                                       # intializes counter for the number of working basis vectors used
    nit      = np.min([ncol, na])                      # number of greedy iterations to perform
    nrhs     = int(np.ceil(ncol/na))                   # max number of RHS in least squares problem
    ncj_min  = int(np.floor(ncol/nit))                 # minimum number of working basis vectors per iteration
    naj_min  = int(np.floor(na*nrhs/ncol))             # minimum number of sample nodes to add per iteration
    full_ind = np.arange(residual_basis.shape[0])      # array of all node inidices

    for j in range(nit):
        ncj = ncj_min            # number of working basis vectors for current iteration
        if j <= (ncol % nit - 1):
            ncj += 1
        naj = naj_min            # number of sample nodes to add during current iteration
        if np.logical_and(nrhs == 1, j<=(na % ncol - 1)):
            naj += 1
        if j == 0:
            R = residual_basis[:, 0:ncj]
        else:
            for q in range(ncj):
                vec     = np.linalg.lstsq(residual_basis[s, 0:nb], residual_basis[s, nb+q-1], rcond=None)[0]
                R[:, q] = residual_basis[:, nb+q-1]-residual_basis[:, 0:nb]@vec
        for k in range(naj):
            # choose node with largest average error
            ind = np.setdiff1d(full_ind, s)
            n   = np.where(np.isin(full_ind, ind))[0][np.argmax(np.sum(R[ind]*R[ind], axis=1))]
            s   = np.append(s, n)
        nb += ncj
        
    return np.sort(s)