import numpy as np
import dill as pickle
import scipy.linalg as la


def compute_svd(
  data,
  energy_min=1e-8,
  n_bases=-1,
  nb_samples=-1,
  get_bases=False,
  save_dir=None,
  verbose=True
):
  """
  Saves left singular vectors and singular values of snapshot and residual data for each subdomain.

  inputs:
  ddmdl: instance of DD_model class
  snapshots: list where snapshots[i] = snapshot vector corresponding to ith parameter set
  residuals: list where residuals[i][j] = Newton residual at jth iteration corresponding to ith parameter set
  save_dir: [optional] directory to save data to. Default is current working directory.
  """
  svd = {}
  for (k, data_k) in data.items():
    svd[k] = []
    for (i, data_ki) in enumerate(data_k):
      if (nb_samples > 0):
        nb_samples_max = len(data_ki)
        nb_samples_ki = min(nb_samples_max, nb_samples)
        indices = np.arange(nb_samples_ki)
        indices = np.random.choice(indices, size=nb_samples_ki, replace=False)
        data_ki = data_ki[indices]
      if verbose:
        print(f"Performing SVD for dataset '{k}-{i+1}' ...")
      svd[k].append(perform_svd(data_ki.T))
  if (save_dir is not None):
    if verbose:
      print("Saving SVD data ...")
    pickle.dump(svd, open(save_dir+"/svd.p", "wb"))
  if get_bases:
    bases = get_bases_from_svd(svd, energy_min=energy_min, n_bases=n_bases)
    if (save_dir is not None):
      if verbose:
        print("Saving POD bases ...")
      pickle.dump(bases, open(save_dir+"/bases.p", "wb"))
    return svd, bases
  else:
    return svd


def get_bases_from_svd(
  svd,
  energy_min=1e-8,
  n_bases=-1
):
  """
  Computes POD bases given saved SVD data.

  inputs:
  data_dict: dictionary with fields
        "left_vecs" : list of matrices of left singular vectors of snapshot data for each subdomain
        "sing_vals" : list of vectors of singular values of snapshot data for each subdomain
  ec: [optional] energy criterior for choosing size of basis. Default is 1e-8
  nbasis: [optional] size of basis. Setting to -1 uses energy criterion. Default is -1.

  output:
  bases: list of matrices containing POD basis for each subdomain
  """
  bases = {}
  for (k, svd_k) in svd.items():
    bases[k] = []
    n_bases_k = n_bases[k] if isinstance(n_bases, dict) else n_bases
    energy_min_k = energy_min[k] if isinstance(energy_min, dict) else energy_min
    for svd_ki in svd_k:
      bases[k].append(
        compute_pod_bases(
          svd=svd_ki, energy_min=energy_min_k, n_bases=n_bases_k
        )
      )
  return bases


def compute_pod_bases(
  svd=None,
  data=None,
  energy_min=1e-5,
  n_bases=-1
):
  """
  Compute POD basis given snapshot data.
  inputs:
    data: (N, M)-array of snapshot data
    ec: [optional] energy criterion for selecting number of basis vectors. Default is 1e-5
    n_bases: [optional] specify number of basis vectors. The default of -1 uses the energy criterion.
         Setting n_bases > 0 overrides the energy criterion.

  outputs:
    U: (N, n_bases)-array containing POD basis
    s: singular values of data
  """
  svd = perform_svd(data) if (svd is None) else svd
  if (n_bases <= 0):
    s_sq = svd["s"]**2
    energy = np.cumsum(s_sq) / np.sum(s_sq)
    n_bases = np.where(energy > 1-energy_min)[0].min()+1
  n_bases = min(svd["u"].shape[0], n_bases)
  return svd["u"][:,:n_bases]


def perform_svd(data):
  svd = la.svd(data, full_matrices=False, check_finite=False)
  return {"u": svd[0], "s": svd[1], "vh": svd[2]}
