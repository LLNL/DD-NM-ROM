#!/bin/bash -i

# =====================================
# Deep Learning @ LLNL
# =====================================
# The present script installs PyTorch on LC TOSS systems

# Inputs
# -------------------------------------
env_file=env.yml
anaconda_dir=/Users/zanardi1/Workspace/Applications/miniconda3
#/collab/usr/gapps/python/${SYS_TYPE}/anaconda3-2024.02
# -------------------------------------

if [ ! -d ${anaconda_dir} ]; then
  # Create installation directory
  mkdir -p ${anaconda_dir} && rm -rf ${anaconda_dir}
  # Install conda
  bash /collab/usr/gapps/python/${SYS_TYPE}/conda/Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -f -p ${anaconda_dir}
fi

source ${anaconda_dir}/bin/activate
conda env create -f ${env_file}
