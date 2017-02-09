#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000000

int main(int argc, char *argv[]) {

	int rank, size;
	int i;
	double A[N];

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank == 2) {
		for(i=0;i<N;i++)
			A[i] = 100;

		// MPI_Bcast(A, N, MPI_DOUBLE, 2, MPI_COMM_WORLD);
	}


	printf("%d: Before A[0] = %f\n", rank, A[0]);

	MPI_Bcast(A, N, MPI_DOUBLE, 2, MPI_COMM_WORLD);

	printf("%d: After A[0] = %f\n", rank, A[0]);

	MPI_Finalize();
}







