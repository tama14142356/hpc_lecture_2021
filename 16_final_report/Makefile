answer: answer.cpp
		mpicxx answer.cpp -fopenmp -march=native -O3
		mpirun -np 4 ./a.out
example: example.cpp
		mpicxx example.cpp -O3
		mpirun -np 4 ./a.out
