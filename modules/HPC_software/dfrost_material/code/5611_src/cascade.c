#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100000

int main(int argc, char *argv[]) {

	int size, rank;
	int src, dest;
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

	dest = (rank+1)%size;
	src = (rank-1+size)%size;
	if(rank == 0)
		src = MPI_PROC_NULL;
	if(rank == size-1)
		dest = MPI_PROC_NULL;

	if(rank%2 == 0) {
		MPI_Send(A, N, MPI_INT, dest, 0, MPI_COMM_WORLD);
		MPI_Recv(B, N, MPI_INT, src, 0, MPI_COMM_WORLD, &stat);
	} else {
		MPI_Recv(B, N, MPI_INT, src, 0, MPI_COMM_WORLD, &stat);
		MPI_Send(A, N, MPI_INT, dest, 0, MPI_COMM_WORLD);
	}

/*
	if(rank == 0) {
		printf("%d: About to send A\n", rank);
		MPI_Send(A, N, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD);
	} else if(rank == size - 1) {
		printf("%d: About to recv B\n", rank);
		MPI_Recv(B, N, MPI_INT, (rank-1+size)%size, 0, MPI_COMM_WORLD, &stat);
	} else {
		printf("%d: About to send A\n", rank);
		MPI_Send(A, N, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD);
		printf("%d: About to recv B\n", rank);
		MPI_Recv(B, N, MPI_INT, (rank-1+size)%size, 0, MPI_COMM_WORLD, &stat);
	}
*/

	printf("%d: Done!\n", rank);

	MPI_Finalize();

	return 0;
}
