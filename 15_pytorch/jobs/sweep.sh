#!/bin/bash
#YBATCH -r am_4
#SBATCH -N 1
#SBATCH -J sweeptest

MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

. /etc/profile.d/modules.sh

module load cuda/11.1
module load nccl/cuda-11.1/2.7.8
module load openmpi

# wandb sweep sweep.yaml
wandb agent tomo/hpc_lecture_2021-15_pytorch/g1w02pv3
