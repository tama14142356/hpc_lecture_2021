# final report 
## for cpu
answer.cpp
## for gpu
my_cuda_mpi.cu

### TSUBAME setup
#### Interactive node
qrsh -g tga-hpc-lecture -l f_node=1 -l h_rt=0:50:00 -ar 予約番号  
#### Job schedule
qsub -g tga-hpc-lecture job.sh "make answer"  
qsub -g tga-hpc-lecture job.sh "mpirun -np 4 ./a.out"  
#### Job monitor (r: 実行中, qw: 順番待ち)
qstat
#### Job delete
qdel ジョブID

#### Modules
echo '' >> ~/.bashrc  
echo '# Modules' >> ~/.bashrc  
echo 'source /etc/profile.d/modules.sh' >> ~/.bashrc  
echo 'module load vim cmake gcc cuda/11.2.146 openmpi nccl cudnn intel' >> ~/.bashrc  
source ~/.bashrc

