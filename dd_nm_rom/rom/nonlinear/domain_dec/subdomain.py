import copy
import numpy as np
import scipy.sparse as sp

from scipy.interpolate import RBFInterpolator
from dd_nm_rom.rom.utils import hyper_red as hr
from dd_nm_rom.ops import sp_diag, map_nested_dict
from dd_nm_rom.fom.domain_dec.subdomain import Subdomain


class SubdomainROM(object):
  '''
  Class for generating a non-hyper-reduced subdomain of the DD NM-ROM for the 2D steady-state Burgers' Equation with Dirichlet BC.

  inputs:
  subdomain: subdomain class of full-order DD autoencoder
  interior_dict: dictionary for interior states with fields
        decoder:    state_dict with decoder parameters
        latent_dim: latent dimension for decoder
        scale:      scaling vector for normalizing data
        ref:        reference vector for shifting data
  interface_dict: dictionary for interface states with fields
        decoder:    state_dict with decoder parameters
        latent_dim: latent dimension for decoder
        scale:      scaling vector for normalizing data
        ref:        reference vector for shifting data
  cmat: constraint matrix

  methods:
  en_interior:  encoder function for interior state
  en_interface:  encoder function for interface state
  de_interior : decoder function for interior state
  de_interface : decoder function for interface state
  res_jac: compute residual its jacobian on the subdomain
  '''

  def __init__(
    self,
    rom_dim,
    subdomain,
    nn_models,
    scaling=1.0,
    cmat=None,
    constraint_type="strong",
    res_bases=None,
    hr_n_samples=1,
    hr_n_edge_samples_ratio=0.75,
    hr_sample_small_ports=False,
    hr_small_ports_dim=5
  ):
    # ROM dimensions
    # -------------
    self.rom_dim = rom_dim
    # FOM subdomain
    # -------------
    self.subdomain = subdomain
    for k in ("ports", "port_to_nodes", "indices", "n_nodes"):
      setattr(self, k, getattr(self.subdomain, k))
    # Autoencoders
    # -------------
    self.nn_models = nn_models
    # Operators/BC
    # -------------
    self.set_ops_bc()
    self.scaling = float(scaling)
    self.cmat = cmat
    self.constraint_type = constraint_type
    # Compose null matrix for interior
    for k in ("interior",):
      self.cmat[k] = self.cmat[k][:,:self.rom_dim[k]]
    # Hyper-reduction (HR)
    # -------------
    self.res_bases = res_bases
    self.hr_n_samples = hr_n_samples
    self.hr_n_edge_samples_ratio = hr_n_edge_samples_ratio
    self.hr_sample_small_ports = hr_sample_small_ports
    self.hr_small_ports_dim = hr_small_ports_dim
    self.set_hr_mode(active=False)
    # Control variabless
    self.hr_init = False
    self.hr_active = False

  # Autoencoders
  # ===================================
  def encode(self, x, element="interior"):
    return self.nn_models[element].encoder(x)

  def decode(self, x, element="interior"):
    return self.nn_models[element].decoder(x)

  # Operators/BC
  # ===================================
  def set_ops_bc(self):
    # matrices for computing interior state residual
    self.bc = map_nested_dict(self.subdomain.bc, copy.copy)
    self.diff_ops = map_nested_dict(self.subdomain.diff_ops, copy.copy)
    self.incl_ops = map_nested_dict(self.subdomain.incl_ops, copy.copy)

  # Hyper-reduction (HR)
  # ===================================
  def set_hr_mode(self, active=False):
    self.hr_active = active
    if self.hr_active:
      if (not self.hr_init):
        self.init_hr_mode()
      self.reconstruct_uv = self._reconstruct_uv_hr
      self.compute_rhs_jac = self._compute_rhs_jac_hr
    else:
      self.reconstruct_uv = self._reconstruct_uv
      self.compute_rhs_jac = self._compute_rhs_jac

  def init_hr_mode(self):
    self.set_dim_hr()
    self.set_indices_hr()
    self.set_ops_bc_hr()
    self.set_decoder_hr()
    self.hr_init = True

  # Number of samples
  # -----------------------------------
  def set_dim_hr(self):
    # Check hyper reduction inputs
    if (self.res_bases is None):
      raise ValueError("Please, provide residual bases for HR.")
    # Compute parameters for hyper reduction
    n_samples_max, self.n_res_bases = self.res_bases.shape
    self.hr_n_samples = np.clip(
      self.hr_n_samples, self.n_res_bases, n_samples_max
    )
    # Set number of nodes on residual edges
    n_edges_max = n_samples_max - 2*self.n_nodes["interior"]
    self.hr_n_edge_samples = self.hr_n_edge_samples_ratio * self.hr_n_samples
    self.hr_n_edge_samples = min(int(self.hr_n_edge_samples), n_edges_max)

  # Indices
  # -----------------------------------
  def set_indices_hr(self):
    # Select residual nodes for hyper reduction
    # -------------
    self.hr_ind_uv = {"res": self.sample_res_nodes_hr()}
    # Split for u and v
    n_nodes_res = self.n_nodes["res"]
    self.hr_ind = {}
    self.hr_ind["res"] = {
      "u": self.hr_ind_uv["res"][self.hr_ind_uv["res"]  < n_nodes_res],
      "v": self.hr_ind_uv["res"][self.hr_ind_uv["res"] >= n_nodes_res] \
           - n_nodes_res
    }
    # Finds nodes for interior and interface states
    # -------------
    # Loop over elements
    for e_k in ("interior", "interface"):
      self.hr_ind[e_k] = {}
      self.hr_ind_uv[e_k] = []
      # Collect matrices
      matrices = [op[e_k] for op in self.diff_ops.values()] \
               + [self.incl_ops[e_k]]
      # Loop over variables
      for x_k in ("u", "v"):
        row_ind = self.hr_ind["res"][x_k]
        col_ind = hr.get_col_indices(row_ind, matrices)
        # Store
        self.hr_ind[e_k][x_k] = col_ind
        self.hr_ind_uv[e_k].append(copy.deepcopy(col_ind))
        if (x_k == "v"):
          self.hr_ind_uv[e_k][-1] += self.n_nodes[e_k]
      self.hr_ind_uv[e_k] = np.concatenate(self.hr_ind_uv[e_k], dtype=np.int32)

  def sample_res_nodes_hr(self):
    '''
    Greedy algorithm to select sample nodes for hyper reduction.

    outputs:
      samples: array of sample nodes
    '''
    # Initialize samples
    samples = np.array([], dtype=np.int32)
    # Include nodes from small ports
    if self.hr_sample_small_ports:
      for p in self.ports:
        nodes = self.port_to_nodes[p]
        if (nodes.size <= self.hr_small_ports_dim):
          inter = np.nonzero(np.isin(self.indices["res"], nodes))[0]
          inter = np.concatenate([inter, inter + self.n_nodes["res"]])
          samples = np.union1d(samples, inter)
    # Greedily sample nodes on residual edges and interior regions
    for k in ("interface", "interior"):
      if (k == "interior"):
        max_samples = self.hr_n_samples
      else:
        max_samples = self.hr_n_edge_samples
      nodes_k = self.subdomain.inter[f"res_{k}"]
      nodes_k = np.concatenate([nodes_k, nodes_k + self.n_nodes["res"]])
      nodes_k = np.setdiff1d(nodes_k, samples)
      indices = hr.select_sample_nodes(
        bases=self.res_bases[nodes_k],
        n_samples=max_samples-len(samples)
      )
      samples = np.union1d(samples, nodes_k[indices])
    return samples

  # Operators
  # -----------------------------------
  def set_ops_bc_hr(self):
    # Differential operators
    # -------------
    # Loop over operators
    self.diff_ops_hr = {}
    for (op_k, op_v) in self.diff_ops.items():
      # Loop over elements
      self.diff_ops_hr[op_k] = {}
      for e_k in ("interior", "interface"):
        # Loop over variables
        self.diff_ops_hr[op_k][e_k] = {}
        for x_k in ("u", "v"):
          submat = np.ix_(self.hr_ind["res"][x_k], self.hr_ind[e_k][x_k])
          self.diff_ops_hr[op_k][e_k][x_k] = op_v[e_k][submat]
    # Inclusion operators
    # -------------
    # Loop over elements
    self.incl_ops_hr = {}
    for e_k in ("interior", "interface"):
      op_v = self.incl_ops[e_k]
      # Loop over variables
      self.incl_ops_hr[e_k] = {}
      for x_i in ("u", "v"):
        for x_j in ("u", "v"):
          submat = np.ix_(self.hr_ind["res"][x_i], self.hr_ind[e_k][x_j])
          self.incl_ops_hr[e_k][x_i+"_"+x_j] = op_v[submat]
    # BC
    # -------------
    # Loop over variables
    self.bc_hr = {}
    for x_k in ("u", "v"):
      self.bc_hr[x_k] = map_nested_dict(
        self.bc[x_k],
        lambda x: x[self.hr_ind["res"][x_k]]
      )

  # Decoders
  # -----------------------------------
  def set_decoder_hr(
    self,
    active=True
  ):
    # Loop over elements
    for e_k in ("interior", "interface"):
      self.nn_models[e_k].set_hr_mode(
        active=active, row_ind=self.hr_ind_uv[e_k]
      )

  # RHS/Jacobian
  # ===================================
  def rhs_jac(
    self,
    x,
    lambdas=None,
    sqp=True
  ):
    """
    Compute residual and its jacobian on subdomain.

    inputs:
    w_interior: (n_interior,) vector - reduced interior state
    w_interface: (n_interface,) vector - reduced interface state
    lam:    (n_constraints,) vector of lagrange multipliers

    outputs:
    res: (nz,) vector - residual on subdomain
    jac: (nz, n_interior + n_interface) array - jacobian of residual w.r.t. (w_interior, w_interface)
    H:   Hessian submatrix for SQP solver
    rhs: RHS block vector in SQP solver
    Ag : constraint matrix times output of interface decoder
    Adg: constraint matrix times jacobian of interface decoder
    """
    # Reconstruct u and v
    uv, dec_jac = self.reconstruct_uv(x)
    # RHS/Jacobian
    rhs, jac = self.compute_rhs_jac(uv, dec_jac)
    # Constraints RHS/Jacobian
    if sqp:
      ck = "interface"
      cjac = copy.deepcopy(self.cmat)
      if (self.constraint_type == "weak"):
        if (self.hr_active):
          self.set_decoder_hr(active=False)
          x_intf, dec_jac_intf = self.decode(x=x[ck], element=ck)
          self.set_decoder_hr(active=True)
        else:
          x_intf = np.concatenate([uv[ck][k] for k in ("u", "v")])
          dec_jac_intf = dec_jac[ck]
        crhs = self.cmat[ck] @ x_intf
        cjac[ck] = cjac[ck] @ dec_jac_intf
      else:
        crhs = self.cmat[ck] @ x[ck]
    # Return
    return Subdomain.s_assemble_sqp(
      rhs=rhs,
      crhs=crhs,
      lambdas=lambdas,
      jac=jac,
      cjac=cjac,
      scaling=self.scaling,
      sqp=sqp
    )

  # HR not active
  # -----------------------------------
  def _reconstruct_uv(
    self,
    x
  ):
    dec_jac = {}
    uv = {e_k: {} for e_k in ("res", "interior", "interface")}
    for e_k in ("interior", "interface"):
      # Reconstruct u and v on 'e_k' section
      uv_k, dec_jac[e_k] = self.decode(x[e_k], element=e_k)
      uv[e_k]["u"] = uv_k[:self.n_nodes[e_k]]
      uv[e_k]["v"] = uv_k[self.n_nodes[e_k]:]
      # Reconstruct u and v on residual section
      for x_k in ("u", "v"):
        if (x_k not in uv["res"]):
          uv["res"][x_k] = 0.0
        uv["res"][x_k] = uv["res"][x_k] + self.incl_ops[e_k] @ uv[e_k][x_k]
    return uv, dec_jac

  def _compute_rhs_jac(
    self,
    uv,
    dec_jac
  ):
    rhs, jac = Subdomain.s_res_jac(
      uv=uv,
      bc=self.bc,
      diff_ops=self.diff_ops,
      incl_ops=self.incl_ops
    )
    for e_k in ("interior", "interface"):
      jac[e_k] = jac[e_k] @ dec_jac[e_k]
    return rhs, jac

  # HR active
  # -----------------------------------
  def _reconstruct_uv_hr(
    self,
    x
  ):
    dec_jac = {}
    uv = {e_k: {} for e_k in ("res", "interior", "interface")}
    for e_k in ("interior", "interface"):
      # Reconstruct u and v on interior/interface section
      uv_k, dec_jac[e_k] = self.decode(x=x[e_k], element=e_k)
      uv[e_k]["u"] = uv_k[:self.hr_ind[e_k]["u"].size]
      uv[e_k]["v"] = uv_k[-self.hr_ind[e_k]["v"].size:]
      # Reconstruct u and v on residual section
      for x_i in ("u","v"):
        for x_j in ("u","v"):
          k = x_i+"_"+x_j
          if (k not in uv["res"]):
            uv["res"][k] = 0.0
          uv["res"][k] = uv["res"][k] + self.incl_ops_hr[e_k][k] @ uv[e_k][x_j]
    return uv, dec_jac

  def _compute_rhs_jac_hr(
    self,
    uv,
    dec_jac
  ):
    # Precompute actions of operators
    diff_ops_uv = Subdomain.s_op_actions(uv, self.diff_ops_hr)
    # RHS and Jacobian
    rhs = Subdomain.s_res(uv, self.bc_hr, diff_ops_uv)
    jac = self._jac_hr(uv, diff_ops_uv)
    for e_k in ("interior", "interface"):
      jac[e_k] = jac[e_k] @ dec_jac[e_k]
    return rhs, jac

  def _jac_hr(
    self,
    uv,
    diff_ops_uv
  ):
    jac = {}
    uv_res = map_nested_dict(uv["res"], sp_diag)
    jac_uu = sp_diag(diff_ops_uv["u"]["Bx"] - self.bc_hr["u"]["x"][0])
    jac_uv = sp_diag(diff_ops_uv["u"]["By"] - self.bc_hr["u"]["y"][0])
    jac_vu = sp_diag(diff_ops_uv["v"]["Bx"] - self.bc_hr["v"]["x"][0])
    jac_vv = sp_diag(diff_ops_uv["v"]["By"] - self.bc_hr["v"]["y"][0])
    for e_k in ("interior", "interface"):
      jac_xx_k = {}
      for x_k in ("u", "v"):
        jac_xx_k[x_k] = uv_res[x_k+"_u"] @ self.diff_ops_hr["Bx"][e_k][x_k] \
                      + uv_res[x_k+"_v"] @ self.diff_ops_hr["By"][e_k][x_k] \
                      + self.diff_ops_hr["Cx"][e_k][x_k] \
                      + self.diff_ops_hr["Cy"][e_k][x_k]
      jac_uu_k = jac_uu @ self.incl_ops_hr[e_k]["u_u"] + jac_xx_k["u"]
      jac_uv_k = jac_uv @ self.incl_ops_hr[e_k]["u_v"]
      jac_vu_k = jac_vu @ self.incl_ops_hr[e_k]["v_u"]
      jac_vv_k = jac_vv @ self.incl_ops_hr[e_k]["v_v"] + jac_xx_k["v"]
      jac[e_k] = sp.bmat(
        [[jac_uu_k, jac_uv_k],
         [jac_vu_k, jac_vv_k]],
        format="csr"
      )
    return jac

  # Initial solution
  # ===================================
  def init_rbf(
    self,
    x,
    mu,
    element,
    smoothing=0.0,
    kernel='linear'
  ):
    if (len(x) != len(mu)):
      raise ValueError(
        "The number of data points (first dimension) in both the input " \
          "parameter space matrix and the FOM solution matrix must match."
      )
    z = np.vstack([self.encode(xi, element=element)[0] for xi in x])
    return RBFInterpolator(y=mu, d=z, smoothing=smoothing, kernel=kernel)
