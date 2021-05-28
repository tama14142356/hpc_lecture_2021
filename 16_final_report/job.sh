#!/bin/bash 
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=0:10:00
#$ -j y
#$ -o output/o.$JOB_ID

source /etc/profile.d/modules.sh
module load gcc intel-mpi 
mpicxx answer.cpp -fopenmp
$1
