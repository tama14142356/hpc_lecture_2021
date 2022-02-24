#!/bin/bash
#$ -cwd
#$ -l rt_F=2
#$ -l USE_SSH=1
#$ -v SSH_PORT=2299
#$ -l h_rt=0:30:00
#$ -N light_dist_test
#$ -j y

START_TIMESTAMP=$(date '+%s')

job_id_base=$JOB_ID
git_root=$(git rev-parse --show-toplevel | head -1)

# ======== Modules ========

source /etc/profile.d/modules.sh

module load cuda/11.1/11.1.1
module load nccl/2.7/2.7.8-1
module load openmpi/3.1.6
module list

# ======== Pyenv ========

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
# pipenv property
export PIPENV_VENV_IN_PROJECT=1
export PIPENV_IGNORE_VIRTUALENVS=1
which python

# ======== MPI ========

# abci-rt_AF
nodes=$NHOSTS
gpus_pernode=4
cpus_pernode=40
gpus=$(($nodes * $gpus_pernode))
cpus=$(($nodes * $cpus_pernode))

# abci
MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_PORT=$((10000 + ($job_id_base % 50000)))

mpi_backend="nccl"
# mpi_backend="mpi"
# mpi_backend="gloo"

# export NCCL_DEBUG=INFO # debug
# export NCCL_IB_DISABLE=1 # tsubame
# export NCCL_IB_TIMEOUT=14

# ======== Scripts ========

mpirun -np $gpus \
    -mca btl self,tcp \
    -npernode $gpus_pernode \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    python "$git_root/15_pytorch/mpi_test_lightening.py"


END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"
END_TIMESTAMP=$(date '+%s')
