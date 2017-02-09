#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 15000

int main(int argc, char *argv[]) {

	int size, rank;
	int i;
	int *A, *B;
	MPI_Status stat;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("Hello world %d of %d\n", rank, size);
	MPI_Barrier(MPI_COMM_WORLD);

	A = malloc(N*sizeof(int));
	B = malloc(N*sizeof(int));

	printf("%d: About to send A\n", rank);
	MPI_Send(A, N, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD);
	printf("%d: About to recv B\n", rank);
	MPI_Recv(B, N, MPI_INT, (rank-1+size)%size, 0, MPI_COMM_WORLD, &stat);

	printf("%d: Done!\n", rank);

	MPI_Finalize();

	return 0;
}
