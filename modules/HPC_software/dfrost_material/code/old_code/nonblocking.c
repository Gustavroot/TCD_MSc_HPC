#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define NUM 100000

int main(int argc, char *argv[]) {


	MPI_Status stat[2];
	MPI_Request req[2];
	int rank, size, i;
	double *A, *B;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	A = (double *)malloc(NUM*sizeof(double));
	B = (double *)malloc(NUM*sizeof(double));
	
	A[0] = rank;
	printf("%d: %f, %f\n", rank, A[0], B[0]);

	MPI_Irecv(B, NUM, MPI_DOUBLE, (rank-1+size)%size, 0, MPI_COMM_WORLD, &req[0]);


	printf("%d: sending to %d\n", rank, (rank+1)%size);
	fflush(stdout);
	MPI_Isend(A, NUM, MPI_DOUBLE, (rank+1)%size, 0, MPI_COMM_WORLD, &req[1]);
	// MPI_Send(A, NUM, MPI_DOUBLE, (rank+1)%size, 0, MPI_COMM_WORLD);

/*
	if(rank == 1) {
		sleep(2);
	}
*/

	printf("%d: getting from %d\n", rank, (rank-1+size)%size);
	fflush(stdout);
	// MPI_Recv(B, NUM, MPI_DOUBLE, (rank-1+size)%size, 0, MPI_COMM_WORLD, &stat);

/*
	for(i=0;i<2;i++) {
		MPI_Wait(&req[i], &stat[i]);
	}
*/
	MPI_Waitall(2, req, stat);

	printf("2nd printf %d: %f, %f\n", rank, A[0], B[0]);

	MPI_Finalize();
}
