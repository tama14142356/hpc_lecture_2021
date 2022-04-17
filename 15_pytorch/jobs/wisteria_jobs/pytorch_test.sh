#!/bin/bash
#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=2
#PJM -L elapse=0:10:00
#PJM -L node-mem=448Gi
#PJM -L proc-core=unlimited
#PJM -g jh160041a
#PJM --mpi proc=16
#PJM --fs /work
#PJM -N mpi_test
#PJM -j
#PJM -X

START_TIMESTAMP=$(date '+%s')

job_id_base=$PJM_JOBID
git_root=$(git rev-parse --show-toplevel | head -1)

# \#"PJM -L rscgrp=share
# \#"PJM -L gpu=4
# ======== Modules ========

module load gcc/8.3.1
module load cuda/11.1
module load pytorch/1.8.1
module load cudnn/8.1.0
module load nccl/2.7.8
module load ompi/4.1.1
# module load ompi-cuda/4.1.1-11.1

# module load cuda/11.2
# module load cudnn/8.1.0
# module load nccl/2.8.4
# module load ompi-cuda/4.1.1-11.2

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
# source "$git_root/15_pytorch/.venv/bin/activate"
which python

# ======== MPI ========

nodes=$PJM_NODE
gpus_pernode=${PJM_PROC_BY_NODE}

gpus=${PJM_MPI_PROC}
# cpus=$(($nodes * $cpus_pernode))
# gpus=$(($nodes * $gpus_pernode))

echo "gpus: $gpus"
echo "gpus per node $gpus_pernode"

# MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_ADDR=$(cat "$PJM_O_NODEINF" | head -1)
MASTER_PORT=$((10000 + ($job_id_base % 50000)))

mpi_backend="nccl"
# mpi_backend="mpi"
# mpi_backend="gloo"

# export NCCL_DEBUG=INFO # debug
# export NCCL_IB_DISABLE=1 # tsubame
# export NCCL_IB_TIMEOUT=14

# ======== Scripts ========

pushd "$git_root/15_pytorch"

set -x

# mpiexec \
#     -n $PJM_MPI_PROC \
#     -npernode $PJM_PROC_BY_NODE \
#     -machinefile $PJM_O_NODEINF \
    # -x NCCL_IB_GID_INDEX=3
mpirun \
    -machinefile $PJM_O_NODEINF \
    -np $PJM_MPI_PROC \
    -npernode $PJM_PROC_BY_NODE \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x NCCL_BUFFSIZE=1048576 \
    python mpi_test.py \
    --mpi_backend "$mpi_backend"
    # --is_mpi4py \

set +x

popd


END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"
END_TIMESTAMP=$(date '+%s')

