import numpy as np
import scipy as sp


def get_col_indices(
  row_ind,
  matrices
):
  '''
  Given HR row-nodes, get indices of n_samples column entries for each matrix in mat_list.

  input:
  row_ind: array/list of row indices
  mat_list: list of matrices to get column indices from

  output:
  indices: column indices
  '''
  matrices = [matrices] if (not isinstance(matrices, list)) else matrices
  col_ind = set()
  for m in matrices:
    m = sp.sparse.coo_matrix(m)
    for row in row_ind:
      col_ind = col_ind.union(set(m.col[m.row==row]))
  return np.sort(np.array(list(col_ind)))

def select_sample_nodes(
  bases,
  n_samples,
  n_bases=-1,
  samples=None
):
  '''
  Greedy algorithm to select sample nodes for hyper reduction.
  inputs:
    bases: (Nr, nr)-array of residual basis vectors
    n_samples: integer of desired number of sample nodes
    n_bases: number of working columns of bases

  outputs:
    samples: array of sample nodes
  '''
  n_bases = bases.shape[1] if (n_bases <= 0) else n_bases
  if (samples is None):
    samples = np.array([], dtype=np.int32)
  # Initialize greedy algorithm
  dn_samples = n_samples - len(samples)                     # number of additional nodes to sample
  n_bases_it = 0                                            # intializes counter for the number of working basis vectors used
  n_it = min(n_bases, dn_samples)                           # number of greedy iterations to perform
  n_rhs = int(np.ceil(n_bases/dn_samples))                  # max number of RHS in least squares problem
  n_bases_it_min = int(np.floor(n_bases/n_it))              # minimum number of working basis vectors per iteration
  dn_samples_min = int(np.floor(dn_samples*n_rhs/n_bases))  # minimum number of sample nodes to add per iteration
  bases_indices = np.arange(bases.shape[0])                 # array of all node inidices
  # Greedy algorithm iterations
  for it in range(n_it):
    # Set number of working basis vectors for current iteration
    nb = n_bases_it_min
    if (it <= (n_bases % n_it - 1)):
      nb += 1
    # Set number of sample nodes to add during current iteration
    dn = dn_samples_min
    if ((n_rhs == 1) and (it <= (dn_samples % n_bases - 1))):
      dn += 1
    if (it == 0):
      r = bases[:,:nb]
    else:
      # Compute least-squares solution
      a = bases[:,:n_bases_it]
      b = bases[:,n_bases_it:n_bases_it+nb]
      x = sp.linalg.lstsq(a[samples], b[samples], cond=None)[0]
      r = b - a@x
    for _ in range(dn):
      # choose node with largest average error
      i = np.setdiff1d(bases_indices, samples)
      n = np.where(np.isin(bases_indices, i))[0]
      n = n[np.argmax(np.sum(r[i]*r[i], axis=1))]
      samples = np.append(samples, n)
    n_bases_it += nb
  # Return samples
  return np.sort(samples)
