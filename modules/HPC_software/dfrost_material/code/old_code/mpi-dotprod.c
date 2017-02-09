#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000

int main(int argc, char *argv[]) {

	int rank, size, i;
	double *a, *b;
	double *pa, *pb;
	double dp;	
	MPI_Status stat;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* Initialize 2 vectors */
	a = malloc(N*sizeof(double));
	b = malloc(N*sizeof(double));
	pa = malloc(N/size*sizeof(double));
	pb = malloc(N/size*sizeof(double));
	
	if(rank == 0) {
		for(i=0;i<N;i++) {
			a[i] = drand48();
			b[i] = drand48();
		}

		/* Calculate DP in serial */
		dp = 0;
		for(i=0;i<N;i++) {
			dp += a[i]*b[i];	
		}
		printf("Serial DP = %f\n", dp);
	}


	/* Split and distribute vectors */
	if(rank == 0) {
		/* Rank 0 can't send to itself */
	for(i=0;i<N/size;i++) {
			pa[i] = a[i]; pb[i] = b[i];
		}
		/* Send bits to everyone else */
		for(i=1;i<size;i++) {
			MPI_Send(&a[i*N/size], N/size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			MPI_Send(&b[i*N/size], N/size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
	} else {
		MPI_Recv(pa, N/size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
		MPI_Recv(pb, N/size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
	}

	/* Calculate partial DPs */

	/* Collect and sum partial DPs */

	/* Win! */

	MPI_Finalize();
	return 0;
}
















