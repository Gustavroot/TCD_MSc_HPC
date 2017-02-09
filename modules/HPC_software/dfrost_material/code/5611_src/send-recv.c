#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100

int main(int argc, char *argv[]) {

	int A[N], B[N];
	int rank, size;
	int i, j;
	MPI_Status stat;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank == 0) {
		for(i=0;i<N;i++) {
			A[i] = i;
		}	
		MPI_Send(A, N, MPI_INT, 1, 0, MPI_COMM_WORLD);
	} else if (rank == 1) {
		// MPI_Recv(B, N, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
	}

	printf("%d: About to finalize\n", rank);
	MPI_Finalize();
}








