import copy
import torch
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

from time import time
from dd_nm_rom import backend as bkd
from dd_nm_rom.solvers import Newton
from dd_nm_rom.ops import map_nested_dict
from dd_nm_rom.fom.domain_dec import DDBurgers2D
from .subdomain import SubdomainROM
from ..autoencoder import AutoencoderNP, MultiAutoencoderNP


class DD_NM_ROM(object):
  '''
  Compute DD NM-ROM for the 2D steady-state Burgers' equation with Dirichlet BC.

  inputs:
  dd_fom: DD model class corresponding to full order DD model.
  intr_net_list: list of paths to trained networks for interior states
  intf_net_list: list of paths to trained networks for interface states
  port_net_list:  [optional] list of paths to trained networks for port states. Required for constraint_type=='strong'
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

  n_constraints: [optional] number of weak constraints for NLP. Default is 1
  constraint_type: [optional] 'weak' or 'strong' port constraints. Default is 'weak'.

  fields:
  subdomain: list of subdomain_LS_ROM or subdomain_LS_ROM_HR classes corresponding to each subdomain, i.e.
         subdomain[i] = reduced subdomain class corresponding to subdomain [i]

  methods:
  FJac: computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver.
  solve: solves for the reduced states of the DD NM-ROM using the Lagrange-Newton-SQP method.
  '''

  def __init__(
    self,
    dd_fom,
    nn_configfiles,
    res_bases=None,
    hr_active=False,
    hr_n_samples=1,
    hr_n_edge_samples_ratio=0.75,
    hr_sample_small_ports=False,
    hr_small_ports_dim=5,
    constraint_type='weak',
    n_constraints_weak=1,
    scaling=1.0,
    seed=None
  ):
    # DD-FOM
    # -------------
    self.dd_fom = dd_fom
    for k in ("nxy", "hxy", "n_subs"):
      setattr(self, k, getattr(self.dd_fom, k))
    # Scaling factor for residual
    self.scaling = self.hxy if (scaling <= 0) else scaling
    # Autoencoders
    # -------------
    self.nn_configs = map_nested_dict(nn_configfiles, torch.load)
    self.nn_models = self.init_nn_models(self.nn_configs)
    # DD-ROM Constraints
    # -------------
    self.seed = seed
    self.constraint_type = constraint_type
    if (self.constraint_type not in ("weak", "strong")):
      raise ValueError(
        f"Could not interpret constraint type: '{self.constraint_type}'. " \
          "Valid options are: ['weak', 'strong']."
      )
    self.n_constraints_weak = n_constraints_weak
    self.compute_rom_dim()
    if (self.constraint_type == "strong"):
      self.set_port_indices()
      self.init_nn_model_intf()
    self.assemble_cmat()
    # HR
    # -------------
    self.res_bases = res_bases
    self.hr_active = hr_active
    self.hr_n_samples = int(hr_n_samples)
    self.hr_n_edge_samples_ratio = np.clip(hr_n_edge_samples_ratio, 0, 1)
    self.hr_sample_small_ports = float(hr_sample_small_ports)
    self.hr_small_ports_dim = int(hr_small_ports_dim)
    # DD-ROM subdomains
    # -------------
    self.subdomains = []
    for (s, sub) in enumerate(self.dd_fom.subdomains):
      inputs_s = {}
      for input_k in ("cmat", "rom_dim", "nn_models"):
        attr_k = getattr(self, input_k)
        inputs_s[input_k] = {
          e_k: attr_k[e_k][s] for e_k in ("interior", "interface")
        }
      self.subdomains.append(
        SubdomainROM(
          subdomain=sub,
          scaling=self.scaling,
          constraint_type=self.constraint_type,
          res_bases=self.res_bases[s],
          hr_n_samples=self.hr_n_samples,
          hr_n_edge_samples_ratio=self.hr_n_edge_samples_ratio,
          hr_sample_small_ports=self.hr_sample_small_ports,
          hr_small_ports_dim=self.hr_small_ports_dim,
          **inputs_s
        )
      )
      if self.hr_active:
        self.subdomains[-1].set_hr_mode(active=True)
    # Initial solution
    # -------------
    self.rbf = None
    # Run time
    # -------------
    self.runtime = 0.0

  # ROM dimensions
  # ===================================
  def compute_rom_dim(self):
    # Read latent dimension of each autoencoder
    self.rom_dim = {}
    for (e_k, cfg_k) in self.nn_configs.items():
      if (e_k not in self.rom_dim):
        self.rom_dim[e_k] = []
      for cfg_ki in cfg_k:
        dim = cfg_ki["decoder"]["latent_dim"]
        self.rom_dim[e_k].append(dim)
    # Compute interface latent dimension form ports ones
    if (self.constraint_type == 'strong'):
      self.rom_dim["interface"] = []
      for sub in self.dd_fom.subdomains:
        dim = 0
        for p in sub.ports:
          dim += self.rom_dim["port"][p]
        self.rom_dim["interface"].append(dim)

  # Port to nodes indices
  # ===================================
  def set_port_indices(self):
    # Check number of port autoencoders
    if (len(self.nn_configs["port"]) != len(self.dd_fom.ports)):
      raise ValueError(
        "The number of port autoencoders doesn't " \
          "match the number of ports available."
      )
    # Set port nodes indices
    self.port_to_nodes = []
    for sub in self.dd_fom.subdomains:
      port_to_nodes_s = {}
      shift = 0
      for p in sub.ports:
        # FOM
        # > Port/interface indices
        port_ind = self.dd_fom.port_to_nodes[p]
        intf_ind = sub.indices["interface"]
        # > Duplicate for u and v
        port_ind = np.concatenate([port_ind, port_ind+self.nxy])
        intf_ind = np.concatenate([intf_ind, intf_ind+self.nxy])
        fom_ind = np.nonzero(np.isin(intf_ind, port_ind))[0]
        # ROM
        port_dim = self.rom_dim["port"][p]
        rom_ind = np.arange(port_dim)+shift
        # Update
        port_to_nodes_s[p] = {"fom": fom_ind, "rom": rom_ind}
        shift += port_dim
      self.port_to_nodes.append(port_to_nodes_s)

  # Autoencoders
  # ===================================
  def init_nn_models(
    self,
    configs
  ):
    # Loop over elements: interior and interface/ports
    nn_models = {}
    for (key_i, cfg_i) in configs.items():
      # Loop over instances in each element
      if (not isinstance(cfg_i, (list, tuple))):
        cfg_i = [cfg_i]
      nn_models[key_i] = [AutoencoderNP(cfg_ij) for cfg_ij in cfg_i]
    return nn_models

  # Interface from ports
  # -----------------------------------
  def init_nn_model_intf(self):
    models = []
    for (s, sub) in enumerate(self.dd_fom.subdomains):
      models.append(
        MultiAutoencoderNP(
          indices=self.port_to_nodes[s],
          input_dim=2*sub.n_nodes["interface"],
          autoencoders={p: self.nn_models["port"][p] for p in sub.ports}
        )
      )
    self.nn_models["interface"] = models

  # Constraint matrices
  # ===================================
  def assemble_cmat(self):
    if (self.constraint_type == "weak"):
      cmat = copy.deepcopy(self.dd_fom.cmat)
      if (self.dd_fom.constraint_type != "weak"):
        cmat, self.n_constraints = DDBurgers2D.s_assemble_cmat_weak(
          cmat=cmat,
          n_constraints_weak=self.n_constraints_weak,
          n_constraints=self.dd_fom.n_constraints,
          seed=self.seed
        )
    else:
      cmat = self._assemble_cmat_strong()
    self.cmat = map_nested_dict(cmat, bkd.to_sparse)

  def _assemble_cmat_strong(self):
    # Compute total number of constraints
    self.n_constraints = 0
    for (p, subs_p) in self.dd_fom.port_to_subs.items():
      port_dim = self.rom_dim["port"][p]
      self.n_constraints += (len(subs_p)-1) * port_dim
    # Assemble constraints matrices
    cmat = {
      "interior": self._init_cmat(element="interior"),
      "interface": self._assemble_cmat_intf()
    }
    return cmat

  def _init_cmat(
    self,
    element
  ):
    cmat = []
    for dim in self.rom_dim[element]:
      cmat.append(sp.coo_matrix((self.n_constraints, dim)))
    return cmat

  def _assemble_cmat_intf(self):
    # Initialize matrices
    cmat = self._init_cmat(element="interface")
    # Fill matrices
    shift = 0
    for (p, subs_p) in self.dd_fom.port_to_subs.items():
      port_dim = self.rom_dim["port"][p]
      for i in range(len(subs_p)-1):
        for (j, l) in enumerate((i,i+1)):
          col = self.port_to_nodes[subs_p[l]][p]["rom"]
          row = np.arange(port_dim) + shift
          data = (-1)**j * np.ones(port_dim)
          cmat[subs_p[l]].col  = np.concatenate((cmat[subs_p[l]].col,  col))
          cmat[subs_p[l]].row  = np.concatenate((cmat[subs_p[l]].row,  row))
          cmat[subs_p[l]].data = np.concatenate((cmat[subs_p[l]].data, data))
        shift += port_dim
    return cmat

  # RHS/Jacobian
  # ===================================
  def rhs_jac(
    self,
    x
  ):
    '''
    Computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver.

    inputs:
    w: vector of all interior and interface states for each subdomain
       and the lagrange multipliers lam in the order
      [intr[0], intf[0], ..., intr[n_subs], intf[n_subs], lam]

    outputs:
    val: RHS of the KKT system
    full_jac: KKT matrix
    runtime: "parallel" runtime to assemble KKT system
    '''
    # Initialize
    # -------------
    start = time()
    rhs, hess, cjac = [], [], []
    crhs = np.zeros(self.n_constraints)
    # > Set Lagrangian multipliers
    lambdas = x[-self.n_constraints:]
    self.runtime += time()-start
    # Loop over subdomains
    # -------------
    si = 0
    runtime_s = 0.0
    for sub in self.subdomains:
      start_s = time()
      z_s = {}
      for e_k in ("interior", "interface"):
        ei = si + sub.rom_dim[e_k]
        z_s[e_k] = x[si:ei]
        si = ei
      # > Compute quantities needed for KKT system
      rhs_s, crhs_s, hess_s, cjac_s = sub.rhs_jac(z_s, lambdas, sqp=True)
      runtime_s = max(time()-start_s, runtime_s)
      # > Store subdomain-related quantities
      start = time()
      rhs.append(rhs_s)
      crhs += crhs_s
      cjac.append(cjac_s)
      hess.append(hess_s)
      self.runtime += time()-start
    self.runtime += runtime_s
    # Assemble
    # -------------
    start = time()
    rhs, jac = DDBurgers2D.s_assemble_rhs_jac(rhs, crhs, cjac, hess)
    self.runtime += time()-start
    return rhs, jac

  # Solution
  # ===================================
  def solve(
    self,
    x0=None,
    mu=None,
    tol=1e-8,
    maxit=50,
    stepsize_min=1e-10,
    verbose=False
  ):
    '''
    Solves for the reduced states of the DD-LS-ROM using the Lagrange-Newton-SQP algorithm.

    inputs:
    w0: initial interior/interface states and lagrange multipliers
    tol: [optional] stopping tolerance for Newton solver. Default is 1e-5
    maxit: [optional] max number of iterations for newton solver. Default is 20
    print_hist: [optional] Boolean to print iteration history for Newton solver. Default is False
    rhs: [optional] Boolean to return RHS vectors for each iteration. Default is False

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
    self.runtime = 0.0
    if (mu is not None):
      x0 = self.init_sol(mu)
    # Solve
    solver = Newton(
      model=self,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      verbose=verbose
    )
    x, rhs, *_ = solver.solve(x0)
    # Assemble solution
    uv, z, lambdas = self.assemble_sol(x, map_on_res=True)
    return uv, z, lambdas, rhs

  def assemble_sol(
    self,
    x,
    map_on_res=False
  ):
    z = {k: [] for k in ("interior", "interface")}
    uv = DDBurgers2D.s_init_uv(size=self.nxy, map_on_res=map_on_res)
    # Loop over subdomains
    si = 0
    for sub in self.subdomains:
      if sub.hr_active:
        sub.set_decoder_hr(active=False)
      # Loop over elements
      for e_k in ("interior", "interface"):
        ei = si + sub.rom_dim[e_k]
        z[e_k].append(x[si:ei])
        uv_i = sub.decode(z[e_k][-1], element=e_k)[0]
        uv = DDBurgers2D.s_extract_uv_sub(
          uv, uv_i, sub, element=e_k, map_on_res=map_on_res
        )
        si = ei
    lambdas = x[-self.n_constraints:]
    return uv, z, lambdas

  def get_ndof(self):
    ndof = 0
    for sub in self.subdomains:
      for e_k in ("interior", "interface"):
        ndof += sub.rom_dim[e_k]
    ndof += self.n_constraints
    return ndof

  def compute_error(
    self,
    uv_fom,
    uv_rom
  ):
    '''
    Compute error between ROM and FOM DD solutions.

    inputs:
    w_intr: list of reduced interior states where
           w_intr[i] = (self.subdomains[i].rom_dim["interior"],) vector of ROM solution interior states
    w_intf: list of reduced interface states where
           w_intf[i] = (self.subdomains[i].rom_dim["interface"],) vector of ROM solution interface states

    u_intr: list of FOM u interior states where
           u_intr[i] = (dd_fom.subdomains[i].n_nodes["interior"],) vector of FOM u solution interior states
    v_intr: list of FOM v interior states where
           v_intr[i] = (dd_fom.subdomains[i].n_nodes["interior"],) vector of FOM v solution interior states
    u_intf: list of FOM u interface states where
           u_intf[i] = (dd_fom.subdomains[i].n_interface,) vector of FOM u solution interface states
    v_intf: list of FOM v interface states where
           v_intf[i] = (dd_fom.subdomains[i].n_interface,) vector of FOM v solution interface states

    output:
    error: square root of mean squared relative error on each subdomain
    '''
    err = 0.0
    for s in range(self.n_subs):
      num_s, den_s = 0.0, 0.0
      for e_k in ("interior", "interface"):
        for x_k in ("u","v"):
          x_fom = uv_fom[e_k][x_k][s]
          x_rom = uv_rom[e_k][x_k][s]
          num_s += np.sum(np.square(x_rom - x_fom))
          den_s += np.sum(np.square(x_fom))
      err += num_s/den_s
    return np.sqrt(err/self.n_subs)

  # Initial solution
  # ===================================
  def init_sol(
    self,
    mu
  ):
    '''
    Computes initial iterate used for solving NM-ROM optimization subproblem.

    inputs:
    mu:    (1, 2) array corresponding to parameters for the 2D Burgers problem.
    mdl:   DD_NM_ROM class

    outputs:
    w0:    interior- and interface- state vector for initial guess to optimization problem
    lam0:  initial guess for lagrange multipliers. computed using least-squares
    runtime: "parallel" timing for generating initial x0 and lam0 iterates
    '''
    start = time()
    if (self.rbf is None):
      raise ValueError("RBF interpolator not initialized.")
    mu = mu.reshape(1,-1)
    z, rhs, jac = [], [], []
    lambdas = np.zeros(self.n_constraints)
    self.runtime += time()-start
    runtime_s = 0.0
    for (s, sub) in enumerate(self.subdomains):
      start_s = time()
      z_s = {}
      for e_k in ("interior", "interface"):
        z_s[e_k] = self.rbf[e_k][s](mu).squeeze()
      rhs_s, *_, cjac_s = sub.rhs_jac(z_s, lambdas, sqp=True)
      runtime_s = max(time()-start_s, runtime_s)
      start = time()
      # Solution
      # -----------
      for e_k in ("interior", "interface"):
        z.append(z_s[e_k])
      # Lambdas
      # -----------
      size = sub.rom_dim["interface"]
      rhs.append(rhs_s[-size:])
      # Jacobian
      jac.append(cjac_s.T[-size:])
      self.runtime += time()-start
    self.runtime += runtime_s
    # Get initial ROM solution and lambdas
    start = time()
    z = np.concatenate(z)
    jac = sp.vstack(jac).toarray()
    rhs = np.concatenate(rhs)
    lambdas = la.lstsq(jac, -rhs)[0]
    x0 = np.concatenate([z, lambdas])
    self.runtime += time()-start
    return x0

  def init_rbf(
    self,
    x,
    mu,
    smoothing=0.0,
    kernel='linear'
  ):
    '''
    inputs:
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
    # Loop over elements
    self.rbf = {}
    for e_k in ("interior", "interface"):
      # Loop over subdomains
      self.rbf[e_k] = []
      for (s, sub) in enumerate(self.subdomains):
        self.rbf[e_k].append(
          sub.init_rbf(
            x=x[e_k][s],
            mu=mu,
            element=e_k,
            smoothing=smoothing,
            kernel=kernel
          )
        )
