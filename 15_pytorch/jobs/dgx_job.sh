#!/bin/bash
#YBATCH -r dgx-a100_4 
#SBATCH -N 1
#SBATCH -J tr_flowe 


export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

source /etc/profile.d/modules.sh
module load cuda/11.1 openmpi nccl/cuda-11.1
mpirun -np 4 python 12_distributed.py
