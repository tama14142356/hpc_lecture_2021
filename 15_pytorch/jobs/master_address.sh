#!/bin/bash
#YBATCH -r am_4 
#SBATCH -N 1 
#SBATCH -J debug_flowe 

MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_PORT=$SLURM_JOBID
mpirun \
    -np 4 \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    python $1
