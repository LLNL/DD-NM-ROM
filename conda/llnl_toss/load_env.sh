#!/bin/bash -i

# =====================================
# Deep Learning @ LLNL
# =====================================
# The present script loads the desired conda environment on LC
# TOSS systems. Add the following function to your ~/.bashrc.

load_conda_env() {
  # Print a help message
  usage() {
    echo -e "Usage: load_conda_env [-h] [-d ANACONDA_DIR] [-n ENV_NAME]" 1>&2
  }

  # Default arguments
  env_name=toss
  anaconda_dir=/collab/usr/gapps/python/${SYS_TYPE}/anaconda3-2024.02

  # Parse arguments
  local OPTIND;
  while getopts ":hd:n:" opt; do
    case ${opt} in
      d)
        anaconda_dir=$OPTARG
        ;;
      n)
        env_name=$OPTARG
        ;;
      h) # Display help.
        usage
        return 0
        ;;
      \?)
        echo -e "Invalid option: $OPTARG" 1>&2
        usage
        return 0
        ;;
      :)
        echo -e "Invalid option: $OPTARG requires an argument" 1>&2
        usage
        return 0
        ;;
    esac
  done
  shift $((OPTIND -1))

  # Activate the environment
  source ${anaconda_dir}/bin/activate
  conda activate ${env_name}
}
