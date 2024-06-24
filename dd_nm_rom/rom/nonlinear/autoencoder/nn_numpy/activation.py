import abc
import copy
import numpy as np

from dd_nm_rom.ops import sp_diag


_ACT_IDS = ("elu", "linear", "mixed", "sigmoid", "swish")

def get(identifier='sigmoid', *args, **kwargs):
  if (isinstance(identifier, str) and (identifier.lower() in _ACT_IDS)):
    return {
      "elu":     ELU,
      "linear":  Linear,
      "mixed":   Mixed,
      "sigmoid": Sigmoid,
      "swish":   Swish
    }[identifier.lower()](*args, **kwargs)
  else:
    raise ValueError(
      f"Could not interpret activation function identifier: '{identifier}'."
    )

# Base activation function
# -------------------------------------
class BaseAct(object):

  def __init__(self):
    self.fun = self._fun
    self.jac = lambda x: sp_diag(self._jac(x))

  def __call__(self, x):
    return self.fun_jac(x)

  def fun_jac(self, x):
    return self.fun(x), self.jac(x)

  @abc.abstractmethod
  def _fun(self, x):
    pass

  @abc.abstractmethod
  def _jac(self, x):
    pass

# Linear
# -------------------------------------
class Linear(BaseAct):

  def _fun(self, x):
    return x

  def _jac(self, x):
    return np.ones_like(x)

# Sigmoid
# -------------------------------------
class Sigmoid(BaseAct):

  def _fun(self, x):
    return 1.0 / (1.0+np.exp(-x))

  def _jac(self, x):
    ex = np.exp(-x)
    return ex / (1.0+ex)**2

# Swish
# -------------------------------------
class Swish(BaseAct):

  def _fun(self, x):
    return x / (1.0 + np.exp(-x))

  def _jac(self, x):
    ex = np.exp(x)
    return ex * (1.0+x+ex) / (1.0+ex)**2

# ELU
# -------------------------------------
class ELU(BaseAct):

  def __init__(self, alpha=1.0):
    self.alpha = float(alpha)
    self._fun = np.vectorize(self._fun)
    self._jac = np.vectorize(self._jac)
    super(ELU, self).__init__()

  def _fun(self, x):
    return x if (x > 0.0) else self.alpha*(np.exp(x)-1.0)

  def _jac(self, x):
    return 1.0 if (x > 0.0) else self.alpha*np.exp(x)

# Mixed
# -------------------------------------
class Mixed(BaseAct):

  def __init__(self, masks):
    super(Mixed, self).__init__()
    self.masks = {}
    for (act, mask) in masks.items():
      self.masks[act] = (get(act), mask.reshape(-1))

  def _fun(self, x):
    y = copy.deepcopy(x)
    for (act, mask) in self.masks.values():
      y[mask] = act._fun(x[mask])
    return y

  def _jac(self, x):
    y = copy.deepcopy(x)
    for (act, mask) in self.masks.values():
      y[mask] = act._jac(x[mask])
    return y
