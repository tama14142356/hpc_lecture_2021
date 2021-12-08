#!/bin/bash

#------ pjsub option --------#
#PJM -L rscgrp=regular-o
#PJM -L node=1
#PJM -L elapse=24:15:00
#PJM -L node-mem=28Gi
#PJM -L proc-core=unlimited
#PJM -g jh160041o
#PJM --fs /work
#PJM -j

#------- Program execution -------#
job_id_base=$PJM_JOBID

START_TIMESTAMP=$(date '+%s')

# source /etc/profile.d/modules.sh
# module load mpi-fftw/3.3.9
module load python/3.8.9
module list

# which pipenv
# source /work/jh160041o/g27030/work_dir/.bash_profile
# source /work/jh160041o/g27030/work_dir/.bashrc
# which pipenv

export LD_PRELOAD=/usr/lib/FJSVtcs/ple/lib64/libpmix.so

git_root=$(git rev-parse --show-toplevel | head -1)
pushd "$git_root/15_pytorch"
source .venv/bin/activate
python 10_cnn.py
popd

END_TIMESTAMP=$(date '+%s')
E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo $E_TIME
# echo "total exec time: $E_TIME s"
# bash "$htime_bash" $E_TIME "total exec time"
