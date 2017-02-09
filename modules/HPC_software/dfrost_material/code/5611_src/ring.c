#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100000

int main(int argc, char *argv[]) {

	int size, rank;
	int i;
	int *A, *B;
	MPI_Status stat;
	MPI_Request req, req2;
	MPI_Request reqs[2];
	MPI_Status stats[2];

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("Hello world %d of %d\n", rank, size);
	MPI_Barrier(MPI_COMM_WORLD);

	A = malloc(N*sizeof(int));
	B = malloc(N*sizeof(int));

	printf("%d: About to send A\n", rank);
	MPI_Isend(A, N, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD, &reqs[0]);
	MPI_Irecv(B, N, MPI_INT, (rank-1+size)%size, 0, MPI_COMM_WORLD, &reqs[1]);
	printf("%d: Both send and recv started now\n", rank);

	sleep(3);
	// MPI_Barrier(MPI_COMM_WORLD);

	
	MPI_Waitall(2, reqs, stats);

	printf("%d: Both send and recv complete now\n", rank);

	printf("%d: Done!\n", rank);

	MPI_Finalize();

	return 0;
}
