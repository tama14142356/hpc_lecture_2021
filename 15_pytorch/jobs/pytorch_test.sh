#!/bin/bash
#YBATCH -r am_2
#SBATCH -N 1
#SBATCH -J disttortest
#SBATCH --output output/torch_dist_test-%j.out

START_TIMESTAMP=$(date '+%s')

job_id_base=$SLURM_JOBID
git_root=$(git rev-parse --show-toplevel | head -1)

# ======== Modules ========

source /etc/profile.d/modules.sh

module load cuda/11.1
module load nccl/cuda-11.1/2.7.8
module load cudnn/cuda-11.1/8.0
module load openmpi/3.1.6
# module load openmpi
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

# ylab
nodes=$SLURM_NNODES
gpus_pernode=2
cpus_pernode=$SLURM_JOB_CPUS_PER_NODE
gpus=$(($nodes * $gpus_pernode))
cpus=$(($nodes * $cpus_pernode))


MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_PORT=$((10000 + ($job_id_base % 50000)))

mpi_backend="nccl"
# mpi_backend="mpi"
# mpi_backend="gloo"

# export NCCL_DEBUG=INFO # debug
# export NCCL_IB_DISABLE=1 # tsubame
# export NCCL_IB_TIMEOUT=14

# ======== Scripts ========

    # -x NCCL_IB_GID_INDEX=3
mpirun -np $gpus \
    -mca btl self,tcp \
    -npernode $gpus_pernode \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    python "$git_root/mpi_test.py" \
    --mpi_backend "$mpi_backend"
    # --is_mpi4py \


END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"
END_TIMESTAMP=$(date '+%s')
