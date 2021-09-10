#!/bin/bash
#$ -cwd
#$ -l rt_G.large=1
#$ -l h_rt=0:10:00
#$ -N ddp
#$ -j y

# MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
# MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

. /etc/profile.d/modules.sh

# module load cuda/11.1
# module load nccl/cuda-11.1/2.7.8
# module load openmpi
module load cuda/11.1/11.1.1
module load nccl/2.7/2.7.8-1
module load openmpi

source ~/.bashrc

mpirun \
    -np 4 \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    python 13_ddp.py
