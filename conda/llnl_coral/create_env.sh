#!/bin/bash -i

# =====================================
# Deep Learning @ LLNL
# =====================================
# The present script installs PyTorch on LC CORAL 1
# systems (i.e., rzansel, lassen, sierra)

# -------------------------------------
# Open-CE with CUDA for Lassen
# -------------------------------------
# See:
#	  https://lc.llnl.gov/confluence/pages/viewpage.action?pageId=785286611
# Note:
#	  Install open-ce in a directory with plenty of free space,
#   like /usr/workspace/${USER}/ or /p/vast1/${USER}/

# Inputs
# -------------------------------------
# Versions
v_cuda=11.8.0
v_opence=1.9.1
v_torch=2.0.1
v_torch_sparse=0.6.17
v_torch_scatter=2.1.1
# Paths
env_name=opence-${v_opence}-cuda-${v_cuda}
anaconda_dir=/usr/workspace/${USER}/Applications/anaconda/${SYS_TYPE}
# -------------------------------------

# Load cuda
module unload cuda && module load cuda/${v_cuda}

if [ ! -d ${anaconda_dir} ]; then
  # Create installation directory
  mkdir -p ${anaconda_dir} && rm -rf ${anaconda_dir}
  # Install conda
  bash /collab/usr/global/tools/opence/${SYS_TYPE}/opence-${v_opence}/Miniconda3-py39_4.12.0-Linux-ppc64le.sh -b -f -p ${anaconda_dir}
fi

# Activate conda environment
source ${anaconda_dir}/bin/activate

# Create an opence environment in conda (Python-3.9)
conda create -y -n ${env_name} python=3.9

# Activate the opence environment
conda activate ${env_name}
export LD_LIBRARY_PATH=${anaconda_dir}/envs/${env_name}/lib:$LD_LIBRARY_PATH

# Register LLNL SSL certificates
conda config --env --set ssl_verify /etc/pki/tls/cert.pem

# Register LC's local channel for Open-CE
lc_channel=/collab/usr/global/tools/opence/${SYS_TYPE}/opence-${v_opence}/condabuild-py3.9-cuda${v_cuda:0:-2}
conda config --env --prepend channels file://${lc_channel}

# Install needed packages
cuda_build=cuda${v_cuda:0:-2}_py39_1
conda install -y pytorch=${v_torch}=${cuda_build}
conda install -y pytorch_sparse=${v_torch_sparse}=${cuda_build}
conda install -y pytorch_scatter=${v_torch_scatter}=${cuda_build}
conda install -y matplotlib scikit-learn pandas h5py
pip install \
  sparselinear \
  dill \
  ipykernel \
  jupyterlab \
  notebook \
  numpy \
  scipy \
  silx \
  tqdm \
  pydoe
