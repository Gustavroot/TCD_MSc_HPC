#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100

int main(int argc, char *argv[]) {

	int rank, size;
	MPI_Status stat;
	int i;
	double *a, *b, dp;
	double *la, *lb, pdp;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank == 0) {
		/* Initialize both vectors */
		a = malloc(N*sizeof(double));	
		b = malloc(N*sizeof(double));	
		for(i=0;i<N;i++) {
			a[i] = drand48();
			b[i] = drand48();
		}
		/* Calc dp in serial */
		dp = 0;
		for(i=0;i<N;i++) {
			dp += a[i]*b[i];
		}
		printf("Serial dp = %f\n", dp);
		
	}

	/* Parallel bit */

	la = malloc(N/size*sizeof(double));
	lb = malloc(N/size*sizeof(double));

	if(rank == 0) {
		/* Send bits of each vector to each other rank */
		for(i=1;i<size;i++) {
			MPI_Send(&a[i*N/size], N/size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			MPI_Send(&b[i*N/size], N/size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
		for(i=0;i<N/size;i++) {
			la[i] = a[i];
			lb[i] = b[i];
		}
	} else {
		/* Recv my bits of vectors from rank 0 */
		MPI_Recv(la, N/size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
		MPI_Recv(lb, N/size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
	}

	/* Calculate my partial dot product */
	pdp = 0;
	for(i=0;i<N/size;i++) {
		pdp += la[i]*lb[i];
	}
	printf("%d: pdp = %f\n", rank, pdp);

	MPI_Barrier(MPI_COMM_WORLD);

	if(rank == 0) {
		/* Get partials from each rank */
		dp = pdp;
		for(i=1;i<size;i++) {
			MPI_Recv(&pdp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &stat);
			dp += pdp;
		}
		
		printf("Parallel dp = %f\n", dp);
		/* Print final sum */
	} else {
		/* Send partial to rank 0 */
		MPI_Send(&pdp, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
}









