answer: answer.cpp
		mpicxx answer.cpp -fopenmp -march=native -O3
		mpirun -np 4 ./a.out
example: example.cpp
		mpicxx example.cpp -O3
		mpirun -np 4 ./a.out
my_cuda_mpi: my_cuda_mpi.cu
	  nvcc my_cuda_mpi.cu -lmpi
		mpirun -np 4 ./a.out
