import torch
import numpy as np
import scipy as sp


# Global
# -------------------------------------
_VALID_BKD = {"torch"}
_VALID_DEVICE = {"cpu", "cuda"}
_VALID_DTYPE = {"float32", "float64"}

# Setting
# -------------------------------------
def set(
  backend="torch",
  device="cpu",
  device_idx=1,
  nb_threads=4,
  epsilon=1e-10,
  floatx="float64"
):
  set_backend(backend)
  set_device(device, device_idx, nb_threads)
  set_floatx(floatx)
  set_epsilon(epsilon)

def get_backend():
  return _BKD

def set_backend(value="torch"):
  global _BKD
  _BKD = value
  if (value not in _VALID_BKD):
    raise ValueError(
      f"Unknown backend: '{value}'. Valid options are: {_VALID_BKD}"
    )

# Conversion
# -------------------------------------
def to_numpy(x):
  if (x is not None):
    if isinstance(x, np.ndarray):
      return x
    elif (torch.is_tensor(x)):
      return x.numpy(force=True)
    elif isinstance(x, (int, float, list, tuple)):
      return np.array(x, dtype=floatx("numpy"))
    else:
      return x

def to_backend(x):
  if (x is not None):
    if (_BKD == "torch"):
      if torch.is_tensor(x):
        return x
      else:
        return torch.as_tensor(to_numpy(x), dtype=floatx("torch"))
    else:
      return to_numpy(x)

def to_sparse(x):
  if sp.sparse.issparse(x):
    return x.tocsr()
  else:
    return sp.sparse.csr_matrix(to_numpy(x))

# Device
# -------------------------------------
def device():
  return _DEVICE

def set_device(value=None, index=0, nb_threads=4):
  if ((value is None) or (value == "cuda")):
    value = "cuda" if torch.cuda.is_available() else "cpu"
  if (value not in _VALID_DEVICE):
    raise ValueError(
      f"Unknown device: '{value}'. Valid options are: {_VALID_DEVICE}"
    )
  if (value == "cuda"):
    value += f":{index}"
  global _DEVICE
  _DEVICE = value
  # Set default device
  try:
    torch.set_default_device(torch.device(_DEVICE))
    torch.set_num_interop_threads(nb_threads)
    torch.set_num_threads(nb_threads)
  except:
    pass

# Epsilon
# -------------------------------------
def machine_eps():
  return float(np.finfo(
    {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]
  ).eps)

def epsilon():
  return _EPSILON

def set_epsilon(value):
  if (value is None):
    value = machine_eps()
  global _EPSILON
  _EPSILON = value

# Float
# -------------------------------------
def floatx(bkd="torch"):
  if (bkd == "torch"):
    return {
      "float16": torch.float16,
      "float32": torch.float32,
      "float64": torch.float64
    }[_FLOATX]
  elif (bkd == "numpy"):
    return {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]
  else:
    return _FLOATX

def set_floatx(value):
  global _FLOATX
  _FLOATX = value
  if (value not in _VALID_DTYPE):
    raise ValueError(
      f"Unknown dtype: '{value}'. Valid options are: {_VALID_DTYPE}"
    )
  try:
    torch.set_default_dtype(floatx())
  except:
    pass
