import collections
import scipy as sp


def sp_diag(x):
  '''
  Helper function to compute sparse diagonal matrix.

  input:
  vec: vector to compute sparse diagonal matrix of

  output:
  D: sparse diagonal matrix with diagonal vec
  '''
  return sp.sparse.spdiags(x, 0, x.size, x.size)

def map_nested_dict(obj, fun):
  if isinstance(obj, collections.Mapping):
    return {k: map_nested_dict(v, fun) for (k, v) in obj.items()}
  else:
    if isinstance(obj, (list, tuple)):
      return [fun(x) for x in obj]
    else:
      return fun(obj)

def face_splitting(A, B):
  '''
  Compute face-splitting product of matrices A and B. Useful in the following identity:
      for matrices A, B and vectors x, y
      hadamard(Ax, By) = face_splitting(A, B) @ kron(x, y).

  It is related to the Khatri-Rao product by
      face_splitting(A, B).T = khatri_rao(A.T, B.T)
  and is equivalent to the row-wise kronecker product of A and B.

  inputs:
      A: (m, n) array
      B: (m, p) array
  outputs:
      C: (m, np) array
  '''
  return sp.linalg.khatri_rao(A.T, B.T).T
