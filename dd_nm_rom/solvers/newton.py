import numpy as np
import scipy.sparse as sp

from time import time
from .basic import Solver


class Newton(Solver):

  def __init__(
    self,
    model=None,
    tol=1e-3,
    maxit=20,
    stepsize_min=1e-10,
    verbose=False
  ):
    super(Newton, self).__init__(
      model=model,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      verbose=verbose
    )

  def solve(
    self,
    x0
  ):
    """
    Solve RHS(x) = 0 using Newton's method.
    inputs:
      rhs_jac: rhsction that returns F(y) and the jacobian of F at y, i.e.
          F, Jac = rhs_jac(y)
      x0: initial guess for Newton's method
      self.tol: [optional] solver tolerance. Default is 1e-10
      self.maxit: [optional] maximum number of iterations. Default is 100
      self.verbose: [optional] Set to True to print iteration history. Default is False

    outputs:
      y: solution of F(y) = 0
      res_vecs: array of residual vectors F(y_i) at each iteration, i.e.
            res_vecs[i] = F(y_i) = residual at ith iteration
      res_hist: residual iteration history
      step_hist: stepsize history
      iter: number of iterations
    """
    # Initialize
    # ---------------
    # > Set first step
    it, x = 0, x0
    rhs, jac, res = self.evaluate(x)
    # > Set histories
    start = time()
    rhs_hist = [rhs]
    res_hist = [res]
    step_hist = [0.0]
    # > Choose a sparse or dense linear solver depending on the problem
    solve = sp.linalg.spsolve if sp.issparse(jac) else np.linalg.solve
    # > Print first step
    if self.verbose:
      self.print_step(it, step_hist[-1], res_hist[-1], header=True)
    self.model.runtime += time()-start
    # Loop until convergence
    # ---------------
    while ((res_hist[-1] >= self.tol) and (it < self.maxit)):
      # > Initialize line search
      start = time()
      dx = solve(jac,-rhs)
      eval_res_tol = lambda stepsize: (1.0 - 2e-4*stepsize)*res_hist[-1]
      self.model.runtime += time()-start
      # > Armijo line search
      x, rhs, jac, res, stepsize = self.line_search(x, dx, eval_res_tol)
      # > Update
      start = time()
      it += 1
      rhs_hist.append(rhs)
      res_hist.append(res)
      step_hist.append(stepsize)
      # > Print step
      if self.verbose:
        self.print_step(it, step_hist[-1], res_hist[-1])
      # > Check step size
      if (stepsize < self.stepsize_min):
        print(f"No stepsize found at iteration {it}.")
        break
      self.model.runtime += time()-start
    start = time()
    # Check convergence
    if (it == self.maxit):
      print(f"Newton solver failed to converge in {self.maxit} iterations.")
    else:
      print(
        f"Newton solver terminated after {it} " \
        f"iterations with residual norm of {res:1.4e}."
      )
    # Return result
    # ---------------
    res = (
      x,
      np.vstack(rhs_hist),
      np.array(res_hist),
      np.array(step_hist),
      it
    )
    self.model.runtime += time()-start
    return res
