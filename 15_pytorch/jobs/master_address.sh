#!/bin/bash
#$ -cwd
#$ -l rt_G.large=1
#$ -l h_rt=2:10:00
#$ -N profile
#$ -j y

START_TIMESTAMP=$(date '+%s')

# MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | cut -d " " -f 6 | cut -d "/" -f 1)
MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
# MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

nodes=$NHOSTS
gpus_pernode=4
gpus=$(($nodes * $gpus_pernode))

# data_root="./data"
###################################################################################################
# copy to local ssd
local_dir="$SGE_LOCALDIR"
# local_dir="$SGE_BEEONDDIR"
rsync -azh "data" "$local_dir/"
data_root="$local_dir/data"
du -d1 -hc --apparent-size "$data_root"
###################################################################################################

. /etc/profile.d/modules.sh

# module load cuda/11.1
# module load nccl/cuda-11.1/2.7.8
# module load openmpi
module load cuda/11.1/11.1.1
module load nccl/2.7/2.7.8-1
module load openmpi

source ~/.bashrc
date_str="$(date '+%Y%m%d_%H%M%S')"
log_file="$JOB_NAME.o$JOB_ID"
log_root="runs/$date_str"

mpirun \
    -np $gpus \
    -npernode $gpus_pernode \
    -x MASTER_ADDR=$MASTER_ADDR \
    -x MASTER_PORT=$MASTER_PORT \
    python 21_profiler.py --path "$data_root" --logs "$log_root/profile_log"

END_TIMESTAMP=$(date '+%s')

E_TIME=$(($END_TIMESTAMP-$START_TIMESTAMP))
echo $E_TIME
mv "$log_file" "$log_root"
