#!/bin/bash
#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=2
#PJM -L elapse=7:00:00
#PJM -L node-mem=448Gi
#PJM -L proc-core=unlimited
#PJM -g jh160041a
#PJM --fs /work
#PJM -N cnn
#PJM -j

# source /etc/profile.d/modules.sh
# module load gcc/8.3.1

module load cuda/11.1
# module load nccl/2.7.8
# module load ompi-cuda/4.1.1-11.1
# # module load pytorch-horovod/1.8.1-0.21.3
# GPUS_PER_NODE=`nvidia-smi -L | wc -l`
# # source $PYTORCH_DIR/bin/activate # ← 仮想環境を activate
source /work/jh160041o/g27030/work_dir/.bash_profile
source /work/jh160041o/g27030/work_dir/.bashrc

# export PYENV_ROOT="$HOME/.pyenv"
# export PATH="$PYENV_ROOT/bin:$PATH"
# eval "$(pyenv init -)"
# eval "$(pyenv virtualenv-init -)"

# module load cuda openmpi nccl cudnn
# $1
python 10_cnn.py
# mpirun -np ${PJM_MPI_PROC} \
        # python 13_ddp.py
