#!/bin/bash 
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=0:10:00
#$ -j y
#$ -o output/o.$JOB_ID

source /etc/profile.d/modules.sh
module load vim cmake gcc cuda/11.2.146 openmpi nccl cudnn intel
mpicxx answer.cpp -fopenmp
$1
