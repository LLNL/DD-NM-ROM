import numpy as np
import scipy as sp

from .basic import Solver


class GaussNewton(Solver):

  def __init__(
    self,
    rhs_jac=None,
    tol=1e-3,
    maxit=20,
    stepsize_min=1e-10,
    verbose=False
  ):
    super(GaussNewton, self).__init__(
      rhs_jac=rhs_jac,
      tol=tol,
      maxit=maxit,
      stepsize_min=stepsize_min,
      verbose=verbose
    )
    self.squared_res = True

  def solve(
    self,
    x0
  ):
    """
    Solve min 0.5*||r(x)||^2 using Gauss-Newton method.

    inputs:
      x0: initial guess for Newton"s method
      tol: [optional] solver tolerance. Default is 1e-10
      self.maxit: [optional] maximum number of iterations. Default is 20
      verbose: [optional] Set to True to print iteration history. Default is False

    outputs:
      x: solution of min ||r(x)||
      conv_hist: convergence iteration history: conv_hist[i] = ||R"r(x_i)||
      step_hist: stepsize history
      it: number of iterations
    """
    # Initialize
    # ---------------
    # > Set first step
    it = 0
    x = x0
    rhs, jac, res = self.evaluate(x)
    # > Set histories
    conv_hist = [np.linalg.norm(jac.T@rhs)]
    step_hist = [0.0]
    # > Print first step
    if self.verbose:
      self.print_step(it, step_hist[-1], conv_hist[-1], header=True)
    # Loop until convergence
    # ---------------
    while ((conv_hist[-1] >= self.tol) & (it < self.self.maxit)):
      # > Initialize line search
      dx, minval = sp.linalg.lstsq(jac,-rhs)[:2]
      eval_res_tol = lambda stepsize: res + 2e-4*stepsize*(minval-res)
      # > Armijo line search
      x, rhs, jac, res, stepsize = self.line_search(x, dx, eval_res_tol)
      # > Update
      it += 1
      conv_hist.append(np.linalg.norm(jac.T@rhs))
      step_hist.append(stepsize)
      # > Print step
      if self.verbose:
        self.print_step(it, step_hist[-1], conv_hist[-1])
      # > Check step size
      if (stepsize < self.stepsize_min):
        print(f"No stepsize found at iteration {it}.")
        break
    # Check convergence
    if (it == self.maxit):
      print(f"Newton solver failed to converge in {self.maxit} iterations.")
    else:
      print(
        f"Gauss-Newton solver terminated after {it} " \
        f"iterations with residual norm of {res:1.4e}."
      )
    # Return result
    # ---------------
    return (
      x,
      np.array(conv_hist),
      np.array(step_hist),
      it
    )
