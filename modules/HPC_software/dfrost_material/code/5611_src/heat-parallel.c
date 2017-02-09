#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 20

int main(int argc, char *argv[]) {

	
	double **A, **B, **tmp;
	int rank, size;
	int i, j, x;
	MPI_Status stat;
	int up, down;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	/* Allocate space for our two grids */
	A = malloc ( (N/size + 2) * sizeof(double *));
	for(i=0;i<N/size+2;i++) 
		A[i] = malloc(N*sizeof(double));

	B = malloc ( (N/size + 2) * sizeof(double *));
	for(i=0;i<N/size+2;i++) 
		B[i] = malloc(N*sizeof(double));

	/* Initialize the two grids */
	if(rank == 0) {
		/* Top side */
		for(i=0;i<N;i++)
			A[1][i] = B[1][i] = 100;
	} else if(rank == size-1) {
		for(i=0;i<N;i++)
			A[N/size][i] = B[N/size][i] = 50;
	}
	for(i=1;i<=N/size;i++) {
		A[i][0] = 25; A[i][N-1] = 75;
	}

	printf("Starting iterations\n");
	MPI_Barrier(MPI_COMM_WORLD);

	for(x=0;x<100;x++) {
		MPI_Barrier(MPI_COMM_WORLD);
		if(rank == 0)
		printf("%d: starting halo exchange %d\n", rank, x);

#ifdef CASCADE
		/* Halo exchange */
		if(rank == 0) {
			/* Top MPI rank */
			MPI_Send(A[5], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
			MPI_Recv(A[6], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &stat);

		} else if (rank == size-1) {
			/* Bottom MPI rank */
			MPI_Recv(A[0], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(A[1], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);

		} else {
			/* Everyone else */
			MPI_Recv(A[0], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(A[5], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);

			MPI_Recv(A[6], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(A[1], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
		}
#else 
		up = rank-1;
		down = rank+1;
		if(rank == 0)
			up = MPI_PROC_NULL;
		if(rank == size-1)
			down = MPI_PROC_NULL;

		printf("Using Sendrecv for HALO\n");
		/* Down comms */
		MPI_Sendrecv(A[5], N, MPI_DOUBLE, down, 0,
				A[0], N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &stat);
		/* Up comms */
		MPI_Sendrecv(A[1], N, MPI_DOUBLE, up, 0,
				A[6], N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &stat);


#endif
	
		if(rank == 0)
		printf("%d: reached end of halo exchange\n", rank);
		MPI_Barrier(MPI_COMM_WORLD);

		/* Update */
		if(rank == 0) {
			/* Top MPI proc */
			for(i=2;i<=N/size;i++) {
				for(j=1;j<N-2;j++) {
					B[i][j] = 0.25*
						(A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
				}
			}

		} else if (rank == size-1) {
			/* Bottom MPI proc */
			for(i=1;i<=N/size-1;i++) {
				for(j=1;j<N-2;j++) {
					B[i][j] = 0.25*
						(A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
				}
			}

		} else {
			/* Everyone else */
			for(i=1;i<=N/size;i++) {
				for(j=1;j<N-2;j++) {
					B[i][j] = 0.25*
						(A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
				}
			}
		}

		/* Swap grids */
		tmp = A;
		A = B;
		B = tmp;
	}

	MPI_Finalize();
}
