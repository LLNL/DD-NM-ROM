import numpy as np


class SubdomainIndices(object):
  """
  Class for generating residual, interior, and interface subdomain indices for a steady-state 2D Burgers FOM.

  inputs:
  monolithic: instance of `Burgers2D` class
  n_sub_x: integer number of subdomains in x direction
  n_sub_y: integer number of subdomains in y direction

  fields:
  rhs:       list, rhs[i] = array of residual indices on subdomain i
  interior:  list, interior[i] = array of interior indices on subdomain i
  interface: list, interface[i] = array of interface indices on subdomain i
  full:      list, full[i] = array of full state (interior and interface) indices on subdomain i
  self.skeleton:  array of indices of self.skeleton, i.e. all interface states for all subdomains
  """

  def __init__(
    self,
    monolithic,
    n_sub_x,
    n_sub_y
  ):
    self.monolithic = monolithic
    for k in ("nx", "ny", "nxy"):
      setattr(self, k, getattr(self.monolithic, k))
    self.n_sub_x = n_sub_x
    self.n_sub_y = n_sub_y
    # Total number of subdomains
    self.n_subs = n_sub_x*n_sub_y
    # Number of x grid points per residual subdomain
    self.nx_sub = self.nx//n_sub_x
    # Number of y grid points per residual subdomain
    self.ny_sub = self.ny//n_sub_y
    # Grid indices
    self.grid = np.arange(self.nxy).reshape(self.ny, self.nx)
    # Set indices
    self.set_res_all()
    self.set_interior_interface()

  def set_res_all(self):
    self.res = [] # Active inner nodes for each subdomain
    self.all = [] # Active inner and ghost nodes for each subdomain (due to FD stencil)
    for j in range(self.n_sub_y):
      for i in range(self.n_sub_x):
        # Define subdomain limits
        xs, xe = self.nx_sub*i, self.nx_sub*(i+1)
        ys, ye = self.ny_sub*j, self.ny_sub*(j+1)
        # Extract subdomain indices from full grid indices
        res_ij = self.grid[ys:ye, xs:xe].flatten()
        # Initialize subdomain "all" indices
        all_ij = set(res_ij)
        # For each row (i.e., node in the grid), finds nonzero columns in the
        # monolithic differential operators leveraging "coo_matrix" format
        for r in res_ij:
          for op in self.monolithic.diff_ops.values():
            op = op.tocoo()
            all_ij = all_ij.union(set(op.col[op.row == r]))
        # Store indices for current subdomain
        self.res.append(np.sort(res_ij))
        self.all.append(np.sort(np.array(list(all_ij))))

  def set_interior_interface(self):
    self.skeleton = set()     # All interface nodes
    self.interior = []        # Interior nodes for each subdomain
    self.interface = []       # Interface nodes for each subdomain
    # Loop over subdomains
    subs = np.arange(self.n_subs)
    for i in subs:
      # Set i-th subdomain
      sub_i = set(self.all[i])
      # Set remaining subdomains
      subs_left = np.delete(subs, i)
      # Define interface nodes for i-th subdomain
      intf_i = set([])
      for j in subs_left:
        # > Take intersection between subdomain i and j
        sub_j = set(self.all[j])
        intf_ij = sub_i.intersection(sub_j)
        intf_i = intf_i.union(intf_ij)
      # Define interior nodes for i-th subdomain
      intr_i = sub_i.difference(intf_i)
      # Store i-th subdomain indices
      self.skeleton = self.skeleton.union(intf_i)
      self.interior.append(np.sort(np.array(list(intr_i))))
      self.interface.append(np.sort(np.array(list(intf_i))))
    self.skeleton = np.array(list(self.skeleton))
