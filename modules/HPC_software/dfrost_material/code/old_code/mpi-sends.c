#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000

int main(int argc, char *argv[]) {

	int rank, size, i;
	double *a, *b;
	double dp;	
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* Initialize 2 vectors */
	a = malloc(N*sizeof(double));
	b = malloc(N*sizeof(double));
	
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


	/* Split and distribute vectors */

	/* Calculate partial DPs */

	/* Collect and sum partial DPs */

	/* Win! */

	MPI_Finalize();
	return 0;
}
