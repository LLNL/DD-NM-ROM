import abc
import numpy as np

from time import time

_PRINT_FMT = "| {0:11d} | {1:11.4e} | {2:11.4e} |"


class Solver(object):

  def __init__(
    self,
    model=None,
    tol=1e-3,
    maxit=20,
    stepsize_min=1e-10,
    verbose=False
  ):
    self.model = model
    self.tol = tol
    self.maxit = maxit
    self.stepsize_min = stepsize_min
    self.verbose = verbose
    self.squared_res = False
    self.set_header()

  def set_header(self):
    self.header = '|   Iteration |    Stepsize |    Residual |\n| '
    n_chars = len(self.header.split('|')[1])-2
    for _ in range(3):
      self.header += '-'*n_chars + ' | '
    self.header = self.header[:-1]

  # Call function
  # ===================================
  def __call__(self, x0):
    return self.solve(x0)

  @abc.abstractmethod
  def solve(self, x0):
    pass

  # Util functions
  # ===================================
  def line_search(
    self,
    x0,
    dx,
    eval_res_tol
  ):
    # Initialize
    # -------------
    start = time()
    stepsize = 1.0
    x = x0 + stepsize*dx
    self.model.runtime += time()-start
    rhs, jac, res = self.evaluate(x)
    # Condition
    # -------------
    start = time()
    cond_fun = lambda res, stepsize: (
      (res >= eval_res_tol(stepsize)) and (stepsize >= self.stepsize_min)
    )
    cond = cond_fun(res, stepsize)
    self.model.runtime += time()-start
    while cond:
      # Update solution
      # -------------
      start = time()
      stepsize *= 0.5
      x = x0 + stepsize*dx
      self.model.runtime += time()-start
      rhs, jac, res = self.evaluate(x)
      # Condition
      # -------------
      start = time()
      cond = cond_fun(res, stepsize)
      self.model.runtime += time()-start
    return x, rhs, jac, res, stepsize

  def evaluate(
    self,
    x
  ):
    rhs, jac = self.model.rhs_jac(x)
    start = time()
    res = np.dot(rhs,rhs)
    if (not self.squared_res):
      res = np.sqrt(res)
    self.model.runtime += time()-start
    return rhs, jac, res

  def print_step(
    self,
    it,
    stepsize,
    res,
    header=False
  ):
    if header:
      print(self.header)
    print(_PRINT_FMT.format(it, stepsize, res))
