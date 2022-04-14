#!/bin/bash

conda_env_root="$1"

if [ -n "$conda_env_root" ]; then
    # first conda setting if need
    mkdir -p "$conda_env_root"

    conda config --add envs_dirs "$conda_env_root/conda_env"
    conda config --add pkgs_dirs "$conda_env_root/conda_pkg"
    # conda config --set changeps1 False
fi

conda create --name wandb-test python=3.8.5

# install pytorh but required cudatoolkit
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
conda install flake8
conda install yapf
conda install jedi
conda install -c conda-forge wandb