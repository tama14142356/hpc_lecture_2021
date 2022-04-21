#!/bin/bash
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=2:10:00
#$ -N mpi_test
#$ -v GPU_COMPUTE_MODE=1 # 1つのGPUを1プロセスのみが利用できるmode
#$ -j y

START_TIMESTAMP=$(date '+%s')

job_id_base=$JOB_ID
git_root=$(git rev-parse --show-toplevel | head -1)

# ======== Modules ========

source /etc/profile.d/modules.sh

module load cuda/11.2.146
# module load cuda/11.0.3
module load cudnn/8.1
module load nccl/2.8.4
# module load openmpi/3.1.4-opa10.10
module load openmpi/3.1.4-opa10.10-t3
# module load openmpi
module list

# ======== Pyenv ========

GROUP_HOME="/gs/hs0/tga-RLA/21M30695"
export PYENV_ROOT="$GROUP_HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
# pipenv property
export PIPENV_VENV_IN_PROJECT=1
export PIPENV_IGNORE_VIRTUALENVS=1
which python

# ======== MPI ========

nodes=$NHOSTS
gpus_pernode=4
cpus_pernode=28
gpus=$(($nodes * $gpus_pernode))
cpus=$(($nodes * $cpus_pernode))

MASTER_ADDR=`head -n 1 $SGE_JOB_SPOOL_DIR/pe_hostfile | cut -d " " -f 1`
MASTER_PORT=$((10000 + ($job_id_base % 50000)))

mpi_backend="nccl"
# mpi_backend="mpi"
# mpi_backend="gloo"

# export NCCL_DEBUG=INFO # debug
export NCCL_IB_DISABLE=1 # tsubame
# export NCCL_IB_TIMEOUT=14

# ======== Scripts ========

set -x

    # -x NCCL_IB_GID_INDEX=3
    # -x PATH \
mpirun -np $gpus \
    -mca btl self,tcp \
    -npernode $gpus_pernode \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    -x PSM2_CUDA=1 \
    -x PSM2_GPUDIRECT=1 \
    -x LD_LIBRARY_PATH \
    python "$git_root/15_pytorch/mpi_test.py" \
    --mpi_backend "$mpi_backend"
    # --is_mpi4py \

set +x

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo "exec time: $E_TIME s"
END_TIMESTAMP=$(date '+%s')
