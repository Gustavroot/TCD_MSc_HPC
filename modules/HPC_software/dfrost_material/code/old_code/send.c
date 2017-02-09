#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000

int main(int argc, char *argv[]) {

	MPI_Status stat;
	int rank, size;
	int i;
	double *A, *B, *C;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	A = calloc(N, sizeof(double));
	B = calloc(N, sizeof(double));
	C = calloc(N, sizeof(double));
	
	if(rank == 0) {
		/* Initialize A */
		for(i=0;i<N;i++) A[i] = 101+i;
		MPI_Send(A, N, MPI_DOUBLE, 1, 2345, MPI_COMM_WORLD);


	} else if(rank == 1) {
		printf("%d: Before recv B[5] = %f\n", rank, B[5]);

		MPI_Recv(B, N, MPI_DOUBLE, MPI_ANY_SOURCE,
			MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

		printf("%d: After recv B[5] = %f\n", rank, B[5]);
	} else {
		printf("%d: I do nothing in this code\n", rank);
	}


	MPI_Finalize();
}
