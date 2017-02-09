#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char *argv[]) {

	MPI_Status stat;
	int rank, size;
	int N = 5000;
	int i;
	double *A, *B;
	double *PA, *PB;
	double dp, pdp;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* Initialize A & B and calculate dp in serial */
	/* Only happens on rank 0 */
	if(rank == 0) {
		A = malloc(N * sizeof(double));
		B = malloc(N * sizeof(double));
		
		for(i=0;i<N;i++) {
			A[i] = 1 - (2 *drand48());
			B[i] = 1 - (2 *drand48());
		}	
		
		dp = 0;
		for(i=0;i<N;i++)
			dp += A[i] * B[i];

		printf("Serial DP = %f\n", dp);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* Distribute bits of A & B to everyone */
	PA = malloc(N/size * sizeof(double));
	PB = malloc(N/size * sizeof(double));

//	if(rank == 0) {
//		for(i=0;i<N/size;i++) {
//			PA[i] = A[i];
//			PB[i] = B[i];
//		for(i=1;i<size;i++) {
//			MPI_Send(&A[i*N/size], N/size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
//			MPI_Send(&B[i*N/size], N/size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
//		}
//	} else {
//		MPI_Recv(PA, N/size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
//		MPI_Recv(PB, N/size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
//	}

	MPI_Scatter(A, N/size, MPI_DOUBLE, PA, N/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(B, N/size, MPI_DOUBLE, PB, N/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	/* On each MPI task, calc partial dot product */
	pdp = 0;
	for(i=0;i<N/size;i++)
		pdp += PA[i] * PB[i];


	/* Everyone send partial dp to rank 0 */
//	dp = 0;
//	if(rank == 0) {
//		dp = pdp;
//
//		for(i=1;i<size;i++) {
//			MPI_Recv(&pdp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &stat);
//			dp += pdp;
//		}
//		
//	} else {
//		MPI_Send(&pdp, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
//	}

	if(rank == 0) {
			MPI_Allreduce(&pdp, &dp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	} else {
			MPI_Allreduce(&XXX, &dp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	}

	/* Rank 0 print total dp and hope it matches serial version */
	printf("%d: Parallel dp = %f\n", rank, dp);


	MPI_Finalize();
	return 0;
}


















