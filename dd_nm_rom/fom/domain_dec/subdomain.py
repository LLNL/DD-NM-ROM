import numpy as np
import scipy.sparse as sp

from dd_nm_rom import backend as bkd
from dd_nm_rom.ops import sp_diag, map_nested_dict


class Subdomain(object):
  """
  Class for generating a subdomain of the DD-FOM for the 2D steady-state Burgers" Equation with Dirichlet BC.

  inputs:
  self.monolithic: instance of Burgers2D class representing full domain problem
  indices["res"]: array of residual indices corresponding to the subdomain to be generated
  indices["interior"]: array of interior indices corresponding to the subdomain to be generated
  indices["interface"]: array of indices["interface"] indices corresponding to the subdomain to be generated
  cmat_intf: constraint matrix corresponding to the interface states of the subdomain
  ports: array containing which ports the subdomain belongs to

  methods:
  set_bc: update boundary condition data on subdomain
  res_jac: compute residual and its jacobian on the subdomain
  """

  def __init__(
    self,
    monolithic,
    indices,
    cmat,
    ports,
    port_to_nodes,
    scaling=1.0
  ):
    self.monolithic = monolithic
    self.scaling = scaling
    self.cmat = cmat
    self.ports = ports
    self.port_to_nodes = port_to_nodes
    # Nodes indices
    self.indices = indices
    self.n_nodes = map_nested_dict(indices, len)
    # Set operators
    self.set_incl_ops()
    self.set_diff_ops()

  def set_incl_ops(self):
    # Intersection between interior/interface and residual nodes
    self.inter = {}
    for e_k in ("interior", "interface"):
      for (ki, kj) in ((e_k, "res"), ("res", e_k)):
        inter_ij = np.isin(self.indices[ki], self.indices[kj])
        self.inter[ki+"_"+kj] = np.nonzero(inter_ij)[0]
    # Inclusion operator for interior/interface and residual nodes
    self.incl_ops = {}
    for e_k in ("interior", "interface"):
      data = np.ones(len(self.inter[e_k+"_res"]))
      shape = (self.n_nodes["res"], self.n_nodes[e_k])
      indices = (self.inter["res_"+e_k], self.inter[e_k+"_res"])
      self.incl_ops[e_k] = sp.csr_matrix((data, indices), shape)

  def set_diff_ops(self):
    self.diff_ops = {}
    for (op_k, op_v) in self.monolithic.diff_ops.items():
      self.diff_ops[op_k] = {}
      for e_k in ("interior", "interface"):
        submat = np.ix_(self.indices["res"], self.indices[e_k])
        self.diff_ops[op_k][e_k] = op_v[submat]

  def set_bc(self):
    """
    Updates boundary condition data on current subdomain

    inputs:
    self.monolithic: instance of Burgers2D class with updated BC data
    """
    self.bc = map_nested_dict(
      self.monolithic.bc,
      lambda x: x[self.indices["res"]]
    )

  def rhs_jac(
    self,
    uv,
    lambdas=None,
    sqp=True
  ):
    """
    Compute residual and its jacobians with respect to interior and interface states.

    inputs:
    u_interior: (n_interior,) vector of u interior states
    v_interior: (n_interior,) vector of v interior states
    u_interface: (n_interface,) vector of u interface states
    v_interface: (n_interface,) vector of  v interface states
    lambdas        : (n_constraints,) vector of lagrange multipliers

    outputs:
    rhs: (2*n_res,) residual vector with u and v residuals concatenated
    jac: (2*n_res, n_interior+n_interface) array - jacobian of residual w.r.t. (w_intr, w_intf)
    H:   Hessian submatrix for SQP solver
    rhs: RHS block vector in SQP solver
    Ax : constraint matrix times interface state

    """
    # Assemble u and v on residual section
    uv = self.map_uv_on_res(uv)
    # RHS and Jacobian
    rhs, jac = Subdomain.s_res_jac(
      uv=uv,
      bc=self.bc,
      diff_ops=self.diff_ops,
      incl_ops=self.incl_ops
    )
    x_intf = np.concatenate([uv["interface"][k] for k in ("u","v")])
    crhs = self.cmat["interface"] @ x_intf
    # Return
    return Subdomain.s_assemble_sqp(
      rhs=rhs,
      crhs=crhs,
      lambdas=lambdas,
      jac=jac,
      cjac=self.cmat,
      scaling=self.scaling,
      sqp=sqp
    )

  def map_uv_on_res(self, uv):
    uv["res"] = {x_k: np.zeros(self.n_nodes["res"]) for x_k in ("u","v")}
    for e_k in ("interior", "interface"):
      res_to_e_k = self.inter["res_"+e_k]
      e_k_to_res = self.inter[e_k+"_res"]
      for x_k in ("u","v"):
        uv["res"][x_k][res_to_e_k] = uv[e_k][x_k][e_k_to_res]
    return uv

  # Static methods
  # ===================================
  @staticmethod
  def s_res_jac(
    uv,
    bc,
    diff_ops,
    incl_ops
  ):
    # Precompute actions of operators
    diff_ops_uv = Subdomain.s_op_actions(uv, diff_ops)
    # RHS and Jacobian
    rhs = Subdomain.s_res(uv, bc, diff_ops_uv)
    jac = Subdomain.s_jac(uv, bc, diff_ops, incl_ops, diff_ops_uv)
    return rhs, jac

  @staticmethod
  def s_op_actions(
    uv,
    diff_ops
  ):
    diff_ops_uv = {}
    for x_k in ("u", "v"):
      diff_ops_uv[x_k] = {}
      for op_k in diff_ops.keys():
        op_v = 0.0
        for e_k in ("interior", "interface"):
          op = diff_ops[op_k][e_k]
          if isinstance(op, dict):
            op = op[x_k]
          op_v = op_v + op @ uv[e_k][x_k]
        diff_ops_uv[x_k][op_k] = op_v
    return diff_ops_uv

  @staticmethod
  def s_res(
    uv,
    bc,
    diff_ops_uv
  ):
    dx = []
    for x_k in ("u", "v"):
      if ("u_u" in uv["res"].keys()):
        u, v = [uv["res"][x_k+"_"+x_i] for x_i in ("u", "v")]
      else:
        u, v = [uv["res"][x_i] for x_i in ("u", "v")]
      dx_k = u*(diff_ops_uv[x_k]["Bx"] - bc[x_k]["x"][0]) \
           + v*(diff_ops_uv[x_k]["By"] - bc[x_k]["y"][0]) \
           + diff_ops_uv[x_k]["Cx"] + bc[x_k]["x"][1] \
           + diff_ops_uv[x_k]["Cy"] + bc[x_k]["y"][1]
      dx.append(dx_k)
    return np.concatenate(dx)

  @staticmethod
  def s_jac(
    uv,
    bc,
    diff_ops,
    incl_ops,
    diff_ops_uv
  ):
    jac = {}
    uv_res = map_nested_dict(uv["res"], sp_diag)
    jac_uu = sp_diag(diff_ops_uv["u"]["Bx"] - bc["u"]["x"][0])
    jac_uv = sp_diag(diff_ops_uv["u"]["By"] - bc["u"]["y"][0])
    jac_vu = sp_diag(diff_ops_uv["v"]["Bx"] - bc["v"]["x"][0])
    jac_vv = sp_diag(diff_ops_uv["v"]["By"] - bc["v"]["y"][0])
    for e_k in ("interior", "interface"):
      jac_xx_k = uv_res["u"] @ diff_ops["Bx"][e_k] \
               + uv_res["v"] @ diff_ops["By"][e_k] \
               + diff_ops["Cx"][e_k] + diff_ops["Cy"][e_k]
      jac_uu_k = jac_uu @ incl_ops[e_k] + jac_xx_k
      jac_uv_k = jac_uv @ incl_ops[e_k]
      jac_vu_k = jac_vu @ incl_ops[e_k]
      jac_vv_k = jac_vv @ incl_ops[e_k] + jac_xx_k
      jac[e_k] = sp.bmat(
        [[jac_uu_k, jac_uv_k],
         [jac_vu_k, jac_vv_k]],
        format="csr"
      )
    return jac

  @staticmethod
  def s_assemble_sqp(
    rhs,
    crhs,
    lambdas,
    jac,
    cjac,
    scaling,
    sqp=True
  ):
    if sqp:
      # To sparse
      jac = map_nested_dict(jac, bkd.to_sparse)
      cjac = map_nested_dict(cjac, bkd.to_sparse)
      # RHS
      rhs = np.concatenate([
        scaling*(jac["interior"].T@rhs),
        scaling*(jac["interface"].T@rhs) + cjac["interface"].T@lambdas
      ])
      # Constraints
      cjac = sp.hstack([cjac["interior"], cjac["interface"]])
      # Hessian
      jac = sp.hstack([jac["interior"], jac["interface"]])
      hess = scaling*(jac.T@jac)
      return rhs, crhs, hess, cjac
    else:
      jac = sp.hstack([jac["interior"], jac["interface"]])
      return rhs, jac
