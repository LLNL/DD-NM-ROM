import numpy as np
import scipy.sparse as sp

from time import time
from .subdomain import Subdomain
from .subdomain_indices import SubdomainIndices
from dd_nm_rom.solvers import Newton


class DDBurgers2D(object):
  """
  Class to compute domain decomposition model from steady-state 2D Burgers FOM

  inputs:
  monolithic: instance of Burgers2D class
  n_sub_x: number of subdomains in x direction. Must divide monolithic.nx
  n_sub_y: number of subdmoains in y direction. Must divide monolithic.ny

  fields:
  nxy:       total number of nodes (finite difference grid points) in full domain model
  n_sub:     number of subdomains
  skeleton:  self.subs_indices of each node in the skeleton
  ports:     list of frozensets corresponding to each port in the DD model.
          ports[i] = frozenset of the subdomains contained in port i
  ports_dict: dictionary of port self.subs_indices where
          ports_dict[port[i]] = self.subs_indices in port[i]
  n_constraints_weak: number of (equality) constraints for DD model
  subdomain: list of instances of Subdomain class where
          subdomain[i] = Subdomain instance corresponding to ith subdomain

  methods:
  set_bc: update boundary condition data
  FJac: computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver
  solve: solves for the states of the DD model using the Lagrange-Newton-SQP method
  """
  def __init__(
    self,
    monolithic,
    n_sub_x,
    n_sub_y,
    n_constraints_weak=1,
    constraint_type="strong",
    scaling=1.0,
    seed=None
  ):
    # FOM monolithic
    # -------------
    self.monolithic = monolithic
    for k in ("nxy", "hxy"):
      setattr(self, k, getattr(self.monolithic, k))
    # Scaling factor for residual
    self.scaling = self.hxy if (scaling <= 0) else scaling
    # DD-FOM subdomains indices
    # -------------
    self.subs_indices = SubdomainIndices(monolithic, n_sub_x, n_sub_y)
    self.n_subs = self.subs_indices.n_subs
    self.set_ports()
    # Constraints
    # -------------
    self.seed = seed
    self.constraint_type = constraint_type
    if (self.constraint_type not in ("weak", "strong")):
      raise ValueError(
        f"Could not interpret constraint type: '{self.constraint_type}'. " \
          "Valid options are: ['weak', 'strong']."
      )
    self.n_constraints_weak = int(n_constraints_weak)
    self.cmat = self.assemble_cmat()
    # DD-FOM subdomains
    # -------------
    self.subdomains = []
    for s in range(self.n_subs):
      cmat_s, indices_s = {}, {}
      for e_k in ("res", "interior", "interface"):
        indices_s[e_k] = getattr(self.subs_indices, e_k)[s]
        if (e_k != "res"):
          cmat_s[e_k] = self.cmat[e_k][s]
      self.subdomains.append(
        Subdomain(
          monolithic=self.monolithic,
          indices=indices_s,
          cmat=cmat_s,
          ports=self.sub_to_ports[s],
          port_to_nodes=self.port_to_nodes,
          scaling=self.scaling
        )
      )
    # Run time
    # -------------
    self.runtime = 0.0

  # Ports
  # ===================================
  def set_ports(self):
    # Assign each interface node to subdomains
    # -------------
    node_intf_to_subs = np.zeros(
      shape=(len(self.subs_indices.skeleton), self.n_subs),
      dtype=bool
    )
    for (i, node_i) in enumerate(self.subs_indices.skeleton):
      for (j, intf_j) in enumerate(self.subs_indices.interface):
        node_intf_to_subs[i,j] = node_i in intf_j
    # Assign subdomains to each port
    # -------------
    subs = np.arange(self.n_subs)
    port_to_subs = set([])
    for mask in node_intf_to_subs:
      port_to_subs.add(frozenset(subs[mask]))
    # > Convert to dictionary
    self.port_to_subs = {}
    for (p, subs_p) in enumerate(list(port_to_subs)):
      self.port_to_subs[p] = np.array(list(subs_p))
    # > List all the ports
    self.ports = np.array(list(self.port_to_subs.keys()))
    # Assign nodes to each port
    # -------------
    self.port_to_nodes = {}
    for (p, subs_p) in self.port_to_subs.items():
      indices = np.zeros(self.n_subs, dtype=bool)
      indices[subs_p] = True
      mask = (node_intf_to_subs == indices).all(axis=1)
      self.port_to_nodes[p] = np.sort(self.subs_indices.skeleton[mask])
    # Assign each port to one or multiple subdomains
    # -------------
    self.sub_to_ports = {}
    for s in range(self.n_subs):
      sub = set([])
      for (p, subs_p) in self.port_to_subs.items():
        if s in subs_p:
          sub.add(p)
      self.sub_to_ports[s] = np.sort(list(sub))

  # Constraint matrices
  # ===================================
  def assemble_cmat(self):
    # Compute total number of constraints
    self.n_constraints = 0
    for (p, subs_p) in self.port_to_subs.items():
      self.n_constraints += (len(subs_p)-1) * len(self.port_to_nodes[p])
    # Assemble constraints matrices
    cmat = {
      "interior": self.init_cmat(element="interior"),
      "interface": self.assemble_cmat_intf()
    }
    # > Make constraint matrices block diagonal for u and v components
    for (e_k, cmat_k) in cmat.items():
      cmat[e_k] = [sp.block_diag([m, m]) for m in cmat_k]
    self.n_constraints *= 2
    # Convert to weak constraints
    if (self.constraint_type == "weak"):
      cmat, self.n_constraints = DDBurgers2D.s_assemble_cmat_weak(
        cmat=cmat,
        n_constraints_weak=self.n_constraints_weak,
        n_constraints=self.n_constraints,
        seed=self.seed
      )
    return cmat

  def init_cmat(self, element):
    cmat = []
    for nodes in getattr(self.subs_indices, element):
      cmat.append(sp.coo_matrix((self.n_constraints, len(nodes))))
    return cmat

  def assemble_cmat_intf(self):
    # Initialize matrices
    cmat = self.init_cmat(element="interface")
    # Fill matrices
    shift = 0
    for (p, subs_p) in self.port_to_subs.items():
      port_nodes = self.port_to_nodes[p]
      port_dim = len(port_nodes)
      for i in range(len(subs_p)-1):
        for (j, l) in enumerate((i,i+1)):
          s = subs_p[l]
          intf_nodes = self.subs_indices.interface[s]
          col = np.where(np.isin(intf_nodes, port_nodes))[0]
          row = np.arange(port_dim) + shift
          dat = (-1)**j * np.ones(port_dim)
          cmat[s].col = np.concatenate((cmat[s].col, col))
          cmat[s].row = np.concatenate((cmat[s].row, row))
          cmat[s].data = np.concatenate((cmat[s].data, dat))
        shift += port_dim
    return cmat

  def set_bc(self):
    """
    Updates boundary condition data on current subdomain

    inputs:
    monolithic: instance of Burgers2D class with updated BC data
    """
    for sub in self.subdomains:
      sub.set_bc()

  def get_ndof(self):
    ndof = 0
    for sub in self.subdomains:
      for e_k in ("interior", "interface"):
        ndof += sub.n_nodes[e_k]
    ndof *= 2
    ndof += self.n_constraints
    return ndof

  def rhs_jac(self, x):
    """
    Computes the KKT system to be solved at each iteration of the Lagrange-Newton SQP solver.

    inputs:
    w: vector of all interior and interface states for each subdomain
       and the lagrange multipliers lambdas in the order
          w = [u_intr[0],
             v_intr[0],
             u_intf[0],
             v_intf[0],
             ...,
             u_intr[n_sub],
             v_intr[n_sub],
             u_intf[n_sub],
             v_intf[n_sub],
             lambdas]

    outputs:
    val: RHS of the KKT system
    full_jac: KKT matrix
    runtime: "parallel" runtime to assemble KKT system

    """
    # Initialize
    # -------------
    start = time()
    rhs, hess, cjac = [], [], []
    crhs = np.zeros(self.n_constraints)
    self.runtime += time()-start
    # Assemble solution
    # -------------
    start = time()
    uv, lambdas = self.assemble_sol(x, map_on_res=False)
    self.runtime += (time()-start) / self.n_subs
    # Loop over subdomains
    # -------------
    runtime_s = 0.0
    for (s, sub) in enumerate(self.subdomains):
      start_s = time()
      # > Get u and v at interior and interface nodes for subdomain 's'
      uv_s = {}
      for e_k in ("interior", "interface"):
        uv_s[e_k] = {}
        for x_k in ("u", "v"):
          uv_s[e_k][x_k] = uv[e_k][x_k][s]
      # > Compute quantities needed for KKT system
      rhs_s, crhs_s, hess_s, cjac_s = sub.rhs_jac(uv_s, lambdas, sqp=True)
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

  def solve(
    self,
    x0=None,
    tol=1e-8,
    maxit=50,
    stepsize_min=1e-10,
    verbose=False
  ):
    """
    Solves for the u and v interior and interface states of the DD-FOM using the Lagrange-Newton-SQP algorithm.

    inputs:
    w0: initial interior/interface states and lagrange multipliers
    tol: [optional] stopping tolerance for Newton solver. Default is 1e-5
    maxit: [optional] max number of iterations for newton solver. Default is 20
    print_hist: [optional] Boolean to print iteration history for Newton solver. Default is False

    outputs:
    u["res"]: u state mapped to full domain
    v["res"]: v state mapped to full domain
    u["interior"]: list of u interior states for each subdomain, i.e
          u["interior"][i] = u interior state on subdomain i
    v["interior"]: list of v interior states for each subdomain, i.e
          v["interior"][i] = v interior state on subdomain i
    u["interface"]: list of u interface states for each subdomain, i.e
           u["interface"][i] = u interface state on subdomain i
    v["interface"]: list of v interface states for each subdomain, i.e
           v["interface"][i] = v interface state on subdomain i
    lambdas:        vector of optimal Lagrange multipliers
    runtime: solve time for Newton solver
    itr: number of iterations for Newton solver
    """
    self.runtime = 0.0
    # Initialize solution
    start = time()
    if (x0 is None):
      x0 = np.zeros(self.get_ndof())
    self.runtime += time()-start
    # Solve
    solver = Newton(
      model=self,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      verbose=verbose
    )
    x, *_ = solver.solve(x0)
    # Assemble solution
    uv, lambdas = self.assemble_sol(x, map_on_res=True)
    return uv, lambdas

  def assemble_sol(self, x, map_on_res=False):
    uv = DDBurgers2D.s_init_uv(size=self.nxy, map_on_res=map_on_res)
    # Loop over subdomains
    si = 0
    for sub in self.subdomains:
      # Loop over elements
      for e_k in ("interior", "interface"):
        ei = si + 2*sub.n_nodes[e_k]
        uv = DDBurgers2D.s_extract_uv_sub(
          uv, x[si:ei], sub, element=e_k, map_on_res=map_on_res
        )
        si = ei
    lambdas = x[-self.n_constraints:]
    return uv, lambdas

  def map_sol_on_elements(
    self,
    solutions,
    map_on_ports=False
  ):
    if (solutions.shape[1] != 2*self.nxy):
      solutions = solutions.T
    data = {}
    for e_k in ("res", "interior", "interface"):
      data[e_k] = []
      for sub in self.subdomains:
        indices = sub.indices[e_k]
        indices = np.concatenate([indices, indices+self.nxy])
        data[e_k].append(solutions[:,indices])
    if map_on_ports:
      data["port"] = []
      for p in self.ports:
        indices = self.port_to_nodes[p]
        indices = np.concatenate([indices, indices+self.nxy])
        data["port"].append(solutions[:,indices])
    return data

  # Static methods
  # ===================================
  @staticmethod
  def s_assemble_cmat_weak(
    cmat,
    n_constraints_weak,
    n_constraints,
    seed=None
  ):
    n_constraints_weak = max(n_constraints_weak, 1)
    n_constraints_weak = min(n_constraints_weak, n_constraints)
    rng = np.random.default_rng(seed)
    rmat = rng.standard_normal((n_constraints_weak, n_constraints))
    for e_k in ("interior", "interface"):
      cmat[e_k] = [sp.csr_matrix(rmat @ m) for m in cmat[e_k]]
    return cmat, n_constraints_weak

  @staticmethod
  def s_init_uv(
    size=1,
    map_on_res=True
  ):
    uv = {}
    for e_k in ("interior", "interface"):
      uv[e_k] = {}
      for x_k in ("u", "v"):
        uv[e_k][x_k] = []
    if map_on_res:
      uv["res"] = {}
      for x_k in ("u", "v"):
        uv["res"][x_k] = np.zeros(size)
    return uv

  @staticmethod
  def s_extract_uv_sub(
    uv,
    uv_i,
    subdomain,
    element,
    map_on_res=True
  ):
    n_nodes = subdomain.n_nodes[element]
    indices = subdomain.indices[element]
    # u velocity
    u = uv_i[:n_nodes]
    uv[element]["u"].append(u)
    if map_on_res:
      uv["res"]["u"][indices] = u
    # v velocity
    v = uv_i[n_nodes:]
    uv[element]["v"].append(v)
    if map_on_res:
      uv["res"]["v"][indices] = v
    return uv

  @staticmethod
  def s_assemble_rhs_jac(
    rhs,
    crhs,
    cjac,
    hess
  ):
    # > RHS
    rhs.append(crhs)
    rhs = np.concatenate(rhs)
    # > Constraints Jacobian
    cjac = sp.hstack(cjac)
    # > Hessians
    hess = sp.block_diag(hess)
    # > Full Jacobian
    jac = sp.bmat(
      [[hess, cjac.T],
       [cjac,   None]],
      format="csr"
    )
    return rhs, jac
