#!/bin/bash
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=0:30:00
#$ -N light_dist_test
#$ --output output/light_dist_test.out
#$ -j y

START_TIMESTAMP=$(date '+%s')

job_id_base=$JOB_ID
git_root=$(git rev-parse --show-toplevel | head -1)

# ======== Modules ========

source /etc/profile.d/modules.sh

#tsubame
module load cuda/11.2.146
module load nccl/2.8.4
module load openmpi/3.1.4-opa10.10
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

# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
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
