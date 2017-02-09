#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <mpi.h>

#define N 20

int size, rank;

void print_grid(double **g) {
	int i, j, k;

	for(k=0;k<size;k++) {	
		if(rank == k) {
			for(i=1;i<6;i++) {
				printf("%d: ", rank);
				for(j=0;j<N;j++) {
					printf("%2.2f\t", g[i][j]);
				}
				printf("\n");
			}
			fflush(stdout);
			usleep(50000);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

int main(int argc, char *argv[]) {
	
	double **A, **B, **t;
	double *tmp;
	int i, j;
	int itcount, itlimit;
	int start_row, end_row;
	int above, below;
	MPI_Status stat;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	/* Allocate 2 grids */
	A = malloc(N*sizeof(double *));
	tmp = malloc(7*N*sizeof(double));
	for(i=0;i<7;i++)
		A[i] = &tmp[i*N];
	
	B = malloc(N*sizeof(double *));
	tmp = malloc(7*N*sizeof(double));
	for(i=0;i<7;i++)
		B[i] = &tmp[i*N];
	
	// printf("%d: malloc succeded\n", rank);

	/* Initialize Boundaries */
	for(i=0;i<N;i++) {
		/* Top and bottom */
		if(rank == 0) {
			A[1][i] = B[1][i] = 100;
		} else if(rank == size-1) {
			A[5][i] = B[5][i] = 100;
		}
	}
	for(i=0;i<7;i++) {
		/* Left and right */
		A[i][0] = A[i][N-1] = 0;
		B[i][0] = B[i][N-1] = 0;
	}

	// printf("%d: init succeded\n", rank);

	print_grid(A);
	itlimit = 1000;
	if(rank == 0) {
		start_row = 2;
		end_row = 5;
		above = MPI_PROC_NULL;
		below = rank+1;
		left = MPI_PROC_NULL;
		right = MPI_PROC_NULL;
	} else if(rank == 3) {
		start_row = 1;
		end_row = 4;
		above = rank-1;
		below = MPI_PROC_NULL;
		left = MPI_PROC_NULL;
		right = MPI_PROC_NULL;
	} else {
		start_row = 1;
		end_row = 5;
		above = rank-1;
		below = rank+1;
		left = MPI_PROC_NULL;
		right = MPI_PROC_NULL;
	}

	MPI_Type_vector(....);
	MPI_Type_commit(coltype);

	for(itcount=0; itcount<itlimit; itcount++) {

		/* Do halo exchange */
		/* Send down */
		if(rank%2 == 0) {
			MPI_Send(A[5], N, MPI_DOUBLE, below, 0, MPI_COMM_WORLD);
			MPI_Recv(A[0], N, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &stat);
		} else
			MPI_Recv(A[0], N, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(A[5], N, MPI_DOUBLE, below, 0, MPI_COMM_WORLD);
		}
		
		MPI_Sendrecv(A[5], N, MPI_DOUBLE, below, 0, 
			A[0], N, MPI_DOUBLE, above, 0, MPI_COMM_WORLD, &stat);
		

		/* Send up */
		MPI_Recv(A[6], N, MPI_DOUBLE, below, 0, MPI_COMM_WORLD, &stat);
		MPI_Send(A[1], N, MPI_DOUBLE, above, 0, MPI_COMM_WORLD);


		/* Send left */
		MPI_Send(&A[0][1], nrows, coltype, ...);
		MPI_Recv(&A[0][ncols-2], nrows, coltype, ...);
		
		/* Send right */
		/* Same-ish as above */
		
		/* Update B from values in A */
		for(i=start_row;i<=end_row;i++)
			for(j=1;j<N-1;j++)
				B[i][j] = 0.25 * (A[i-1][j] + A[i+1][j] +
						A[i][j-1] + A[i][j+1] );


		/* Swap A and B */
		t = A;
		A = B;
		B = t;
	}

	print_grid(A);

	MPI_Finalize();
	
}




















