#!/bin/bash
#$ -cwd
#$ -l rt_G.small=1
#$ -l h_rt=0:10:00
#$ -N ddp
#$ -j y

source /etc/profile.d/modules.sh
module load cuda/11.1/11.1.1
# module load nccl/2.7/2.7.8-1
# module load openmpi

source ~/.bashrc

# export PYENV_ROOT="$HOME/.pyenv"
# export PATH="$PYENV_ROOT/bin:$PATH"
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"

# module load cuda openmpi nccl cudnn
# $1
python 10_cnn.py
