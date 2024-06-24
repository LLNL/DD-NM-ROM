import torch
import numpy as np
import dill as pickle
import scipy.linalg as la
import scipy.sparse as sp

from copy import copy
from dd_nm_rom import solvers
from ..linear.monolithic import select_sample_nodes


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
  b1 = state_dict['net.0.bias'].to('cpu').detach().numpy()

  if 'net.0.weight_mask' in state_dict:
    mask = state_dict['net.0.weight_mask'].to('cpu').detach().numpy()
    W1 = sp.csr_matrix(state_dict['net.0.weight_orig'].to('cpu').detach().numpy()*mask)
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
  return a, da

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

  a   = z*sig
  da  = sig + z*dsig
  return a, da

class monolithic_NMROM:
  '''
  Class for generating a non-hyper-reduced subdomain of the DD NM-ROM for the 2D steady-state Burgers' Equation with Dirichlet BC.

  inputs:
  subdomain: subdomain class of full-order DD model
  net_dict:   dictionary with fields
        decoder:    state_dict with decoder parameters
        latent_dim: latent dimension for decoder
        scale:      scaling vector for normalizing data
        ref:        reference vector for shifting data

  methods:
  encoder: encoder function for interior state
  decoder: decoder function for interface state
  res_jac: compute residual its jacobian on the subdomain
  solve:   solve NMROM
  '''

  def __init__(self, fom, net_dict):
    self.nxy = fom.nxy
    self.latent_dim = net_dict['latent_dim']

    if 'train_time' in net_dict:
      self.train_time = net_dict['train_time']

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {self.device}")

    if net_dict['act_type'] == 'Sigmoid':
      self.act_fun = sigmoid
    elif net_dict['act_type'] == 'Swish':
      self.act_fun = swish

    # encoder weights and biases
    self.en_mask = net_dict['mask'].T
    self.en_W1, self.en_b1, self.en_W2 = get_net_np_params(net_dict['encoder'], self.en_mask.shape)
    self.scale = net_dict['scale'].to('cpu').detach().numpy()
    self.ref   = net_dict['ref'].to('cpu').detach().numpy()

    # interior decoder weights and biases
    self.de_mask = net_dict['mask']
    self.de_W1, self.de_b1, de_W2= get_net_np_params(net_dict['decoder'], self.de_mask.shape)
    self.de_W2  = sp_diag(self.scale)@de_W2

    # matrices for computing interior state residual
    self.Bx = copy(fom.Bx)
    self.By = copy(fom.By)
    self.Cx = copy(fom.Cx)
    self.Cy = copy(fom.Cy)

    self.b_ux1 = copy(fom.b_ux1)
    self.b_uy1 = copy(fom.b_uy1)
    self.b_ux2 = copy(fom.b_ux2)
    self.b_uy2 = copy(fom.b_uy2)

    self.b_vx1 = copy(fom.b_vx1)
    self.b_vy1 = copy(fom.b_vy1)
    self.b_vx2 = copy(fom.b_vx2)
    self.b_vy2 = copy(fom.b_vy2)

  def encoder(self, w):
    '''
    Computes output of encoder

    input:
      w: (latent_dim,) vector
    output:
      out: output of encoder
    '''

    z1  = self.en_W1@((w-self.ref)/self.scale) + self.en_b1
    a1, da1 = self.act_fun(z1)

    out = self.en_W2@a1
    return out

  def decoder(self, w):
    '''
    Computes output of decoder and jacobian with respect to inputs

    input:
      w: (latent_dim,) vector
    output:
      out: output of decoder
      jac: jacobian of output
    '''
    z1  = self.de_W1@w + self.de_b1
    a1, da1 = self.act_fun(z1)
    out = self.de_W2@a1 + self.ref
    jac = self.de_W2@sp_diag(da1)@self.de_W1
    return out, jac

  def res_jac(self, w):
    '''
    Compute residual and its jacobian on subdomain.

    inputs:
    w: (latent_dim,) vector - reduced state

    outputs:
    res: (nz,) vector - residual on subdomain
    jac: (nz, n_interior + n_interface) array - jacobian of residual w.r.t. (w_intr, w_intf)
    '''

    uv, de_jac = self.decoder(w)
    u = uv[:self.nxy]
    v = uv[self.nxy:]

    # compute matrix-vector products on interior and interface
    Bxu = self.Bx@u
    Byu = self.By@u
    Cxu = self.Cx@u
    Cyu = self.Cy@u

    Bxv = self.Bx@v
    Byv = self.By@v
    Cxv = self.Cx@v
    Cyv = self.Cy@v

    # computes u and v residuals
    res_u = u*(Bxu - self.b_ux1) + v*(Byu - self.b_uy1) \
        + Cxu + self.b_ux2 + Cyu + self.b_uy2
    res_v = u*(Bxv - self.b_vx1) + v*(Byv - self.b_vy1) \
        + Cxv + self.b_vx2 + Cyv + self.b_vy2
    res = np.concatenate((res_u, res_v))

    # precompute reused quantities
    diagU = sp_diag(u)
    diagV = sp_diag(v)
    diagU_Bx = diagU@self.Bx
    diagV_By = diagV@self.By

    # jacobian with respect to interior states
    jac_uu = sp_diag(Bxu - self.b_ux1) + diagU_Bx + diagV_By + self.Cx + self.Cy
    jac_uv = sp_diag(Byu - self.b_uy1)
    jac_vu = sp_diag(Bxv - self.b_vx1)
    jac_vv = diagU_Bx + sp_diag(Byv-self.b_vx1) + diagV_By + self.Cx + self.Cy
    jac = sp.bmat([[jac_uu, jac_uv],[jac_vu, jac_vv]])@de_jac

    return res, jac

  def solve(self, w0, tol=1e-6, maxit=20, print_hist=False):
    w, conv_hist, step_hist, iter = solvers.gauss_newton(self.res_jac, w0, tol=tol, maxit=maxit, print_hist=print_hist)
    return w, conv_hist, step_hist, iter

class monolithic_NMROM_HR:
  '''
  Class for generating a hyper-reduced NM-ROM for the 2D steady-state Burgers' Equation with Dirichlet BC.

  inputs:
  subdomain: subdomain class of full-order DD model
  net_dict:   dictionary with fields
        decoder:    state_dict with decoder parameters
        latent_dim: latent dimension for decoder
        scale:      scaling vector for normalizing data
        ref:        reference vector for shifting data
  sample_ratio: [optional] ratio of number of hyper-reduction samples to residual basis size. Default is 2
  n_samples:    [optional] specify number of hyper reduction sample nodes. If n_samples<0, the number of samples is determined by the
          sample ratio. Default is -1.

  methods:
  encoder: encoder function for interior state
  decoder: decoder function for interface state
  res_jac: compute residual its jacobian on the subdomain
  solve:   solve NMROM
  '''

  def __init__(self, fom, net_dict, residual_basis, sample_ratio=2, n_samples=-1):
    self.nxy = fom.nxy
    self.latent_dim = net_dict['latent_dim']

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {self.device}")

    if 'train_time' in net_dict:
      self.train_time = net_dict['train_time']

    if net_dict['act_type'] == 'Sigmoid':
      self.act_fun  = sigmoid
    elif net_dict['act_type'] == 'Swish':
      self.act_fun  = swish

    # select HR nodes
    self.n_hr_nodes = sample_ratio*residual_basis.shape[1] if n_samples<0 else n_samples
    self.hr_ind   = select_sample_nodes(residual_basis, self.n_hr_nodes)
    self.hr_row_u = self.hr_ind[self.hr_ind < self.nxy]
    self.hr_row_v = self.hr_ind[self.hr_ind >= self.nxy] - self.nxy

    # finds nodes for interior and interface states
    self.hr_col_u  = get_hr_col_ind(self.hr_row_u, [fom.Bx, fom.By, fom.Cx, fom.Cy])
    self.hr_col_v  = get_hr_col_ind(self.hr_row_v, [fom.Bx, fom.By, fom.Cx, fom.Cy])
    self.hr_col_uv = np.concatenate([self.hr_col_u, self.hr_col_v+self.nxy]).astype(int)

    # matrices for computing state residual
    I = sp.eye(fom.nxy).tocsr()
    submatrix_indices = np.ix_(self.hr_row_u, self.hr_col_u)
    self.Bx_u = fom.Bx[submatrix_indices]
    self.By_u = fom.By[submatrix_indices]
    self.Cx_u = fom.Cx[submatrix_indices]
    self.Cy_u = fom.Cy[submatrix_indices]
    self.uu_incl = I[submatrix_indices]

    submatrix_indices = np.ix_(self.hr_row_v, self.hr_col_v)
    self.Bx_v = fom.Bx[submatrix_indices]
    self.By_v = fom.By[submatrix_indices]
    self.Cx_v = fom.Cx[submatrix_indices]
    self.Cy_v = fom.Cy[submatrix_indices]
    self.vv_incl = I[submatrix_indices]

    self.b_ux1 = fom.b_ux1[self.hr_row_u]
    self.b_uy1 = fom.b_uy1[self.hr_row_u]
    self.b_ux2 = fom.b_ux2[self.hr_row_u]
    self.b_uy2 = fom.b_uy2[self.hr_row_u]

    self.b_vx1 = fom.b_vx1[self.hr_row_v]
    self.b_vy1 = fom.b_vy1[self.hr_row_v]
    self.b_vx2 = fom.b_vx2[self.hr_row_v]
    self.b_vy2 = fom.b_vy2[self.hr_row_v]

    self.uv_incl = I[np.ix_(self.hr_row_u, self.hr_col_v)]
    self.vu_incl = I[np.ix_(self.hr_row_v, self.hr_col_u)]

    # encoder weights and biases
    self.en_mask = net_dict['mask'].T
    self.en_W1, self.en_b1, self.en_W2 = get_net_np_params(net_dict['encoder'], self.en_mask.shape)
    self.scale = net_dict['scale'].to('cpu').detach().numpy()
    self.ref   = net_dict['ref'].to('cpu').detach().numpy()

    # decoder weights and biases
    self.de_mask = net_dict['mask']
    self.de_W1, self.de_b1, de_W2= get_net_np_params(net_dict['decoder'], self.de_mask.shape)
    self.de_W2  = sp_diag(self.scale)@de_W2

    # decoder subnet weights and biases
    self.de_W1_hr, self.de_b1_hr, de_W2_hr, scale_hr, self.ref_hr, ind = find_subnet(self.hr_col_uv,
                                             self.de_W1,
                                             self.de_b1,
                                             de_W2,
                                             self.scale,
                                             self.ref)
    self.de_W2_hr = sp_diag(scale_hr)@de_W2_hr

  def encoder(self, w):
    '''
    Computes output of encoder

    input:
      w: (latent_dim,) vector
    output:
      out: output of encoder
    '''

    z1  = self.en_W1@((w-self.ref)/self.scale) + self.en_b1
    a1, da1 = self.act_fun(z1)

    out = self.en_W2@a1
    return out

  def decoder(self, w):
    '''
    Computes output of decoder and jacobian with respect to inputs

    input:
      w: (latent_dim,) vector
    output:
      out: output of decoder
      jac: jacobian of output
    '''
    z1  = self.de_W1@w + self.de_b1
    a1, da1 = self.act_fun(z1)
    out = self.de_W2@a1 + self.ref
    jac = self.de_W2@sp_diag(da1)@self.de_W1
    return out, jac

  def decoder_hr(self, w):
    '''
    Computes output of subnet decoder and jacobian with respect to inputs

    input:
      w: (latent_dim,) vector
    output:
      out: output of decoder
      jac: jacobian of output
    '''
    z1  = self.de_W1_hr@w + self.de_b1_hr
    a1, da1 = self.act_fun(z1)
    out = self.de_W2_hr@a1 + self.ref_hr
    jac = self.de_W2_hr@sp_diag(da1)@self.de_W1_hr
    return out, jac

  def res_jac(self, w):
    '''
    Compute residual and its jacobian on subdomain.

    inputs:
    w: (latent_dim,) vector - reduced state

    outputs:
    res: (nz,) vector - residual on subdomain
    jac: (nz, n_interior + n_interface) array - jacobian of residual w.r.t. (w_intr, w_intf)
    '''

    uv, de_jac = self.decoder_hr(w)
    u = uv[:self.hr_col_u.size]
    v = uv[self.hr_col_u.size:]

    uu_res = self.uu_incl@u
    uv_res = self.uv_incl@v
    vu_res = self.vu_incl@u
    vv_res = self.vv_incl@v


    # compute matrix-vector products on interior and interface
    Bxu = self.Bx_u@u
    Byu = self.By_u@u
    Cxu = self.Cx_u@u
    Cyu = self.Cy_u@u

    Bxv = self.Bx_v@v
    Byv = self.By_v@v
    Cxv = self.Cx_v@v
    Cyv = self.Cy_v@v

    # computes u and v residuals
    res_u = uu_res*(Bxu - self.b_ux1) + uv_res*(Byu - self.b_uy1) \
        + Cxu + self.b_ux2 + Cyu + self.b_uy2
    res_v = vu_res*(Bxv - self.b_vx1) + vv_res*(Byv - self.b_vy1) \
        + Cxv + self.b_vx2 + Cyv + self.b_vy2
    res = np.concatenate((res_u, res_v))

    # jacobian with respect to interior states
    jac_uu = sp_diag(Bxu-self.b_ux1)@self.uu_incl \
         + sp_diag(uu_res)@self.Bx_u \
         + sp_diag(uv_res)@self.By_u + self.Cx_u + self.Cy_u
    jac_uv = sp_diag(Byu-self.b_uy1)@self.uv_incl
    jac_vu = sp_diag(Bxv-self.b_vx1)@self.vu_incl
    jac_vv = sp_diag(vu_res)@self.Bx_v \
         + sp_diag(Byv-self.b_vx1)@self.vv_incl \
         + sp_diag(vv_res)@self.By_v + self.Cx_v + self.Cy_v
    jac = sp.bmat([[jac_uu, jac_uv],[jac_vu, jac_vv]])@de_jac

    return res, jac

  def solve(self, w0, tol=1e-6, maxit=20, print_hist=False):
    w, conv_hist, step_hist, iter = solvers.gauss_newton(self.res_jac, w0, tol=tol, maxit=maxit, print_hist=print_hist)
    return w, conv_hist, step_hist, iter