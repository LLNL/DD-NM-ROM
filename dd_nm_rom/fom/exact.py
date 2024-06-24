import numpy as np


class Burgers2DExact(object):

  def set_params(self, a1, lam, viscosity):
    self.a1 = a1
    self.lam = lam
    self.viscosity = viscosity

  def u(self, x, y):
    return -2.0 * self.viscosity \
      * (self.a1 + self.lam*(np.exp(self.lam*(x-1.0)) \
      - np.exp(-self.lam*(x-1.0)))*np.cos(self.lam*y)) / self.phi(x, y)

  def v(self, x, y):
    return 2.0 * self.viscosity \
      * (self.lam*(np.exp(self.lam*(x-1.0)) \
      + np.exp(-self.lam*(x-1.0)))*np.sin(self.lam*y)) / self.phi(x, y)

  def phi(self, x, y):
    return self.a1 + self.a1*x \
      + (np.exp(self.lam*(x-1.0)) + np.exp(-self.lam*(x-1.0))) \
      * np.cos(self.lam*y)
