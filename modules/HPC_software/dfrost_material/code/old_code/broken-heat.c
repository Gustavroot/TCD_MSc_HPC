#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

#define N 20
int rank, size;
int numblocks;

void print_grid(double **X) {

	int i, j, k;
	if(rank ==0 ) {
		printf("--------------------------------------------------------\n");
	}

	for(k=0;k<size;k++) {
		if(rank == k) {

			for(i=1;i<=N/numblocks;i++) {
				printf("%d: ", rank);
				for(j=0;j<=N/2;j++) {
					printf("%2.2f\t", X[i][j]);
				}
				printf("\n");
			}
		}
		fflush(stdout);
		usleep(1000);
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

int main(int argc, char *argv[]) {

	int i, j, k;
	int srow, frow;
	double **g1, **g2, **tmpgrid;
	double *tmp;
	int up, down;
	MPI_Status stat[4];
	MPI_Request req[4];


	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	numblocks = size/2;	

	/* Allocate space for two grids */
	g1 = malloc( (N/numblocks + 2) * sizeof(double *));
	tmp = malloc( (N/numblocks + 2) * (N/2+1) * sizeof(double));
	for(i=0;i<N/numblocks+2;i++) {
		g1[i] = &tmp[(N/2+1) * i];
	}
	
	g2 = malloc( (N/numblocks + 2) * sizeof(double *));
	tmp = malloc( (N/numblocks + 2) * (N/2+1) * sizeof(double));
	for(i=0;i<N/numblocks+2;i++) {
		g2[i] = &tmp[(N/2+1) * i];
	}
	
	/* Initialize grids */	
	/* Top row set to 100 */
	if(rank == 0 || rank == 1) {
		for(i=0;i<=N/2;i++) {
			g1[1][i] = g2[1][i] = 100;
		}
	}

	/* Bottom row set to 200 */
	if(rank == size-2 || rank == size-1) {
		for(i=0;i<=N/2;i++) {
			g1[N/numblocks][i] = g2[N/numblocks][i] = 200;
		}
	}

	/* Set left column to 50 */
	if(rank%2 == 0) {
		for(i=0;i<N/numblocks+2;i++) {
			g1[i][0] = g2[i][0] = 50;
		}
	}

	/* Set right colum to 75 */
	if(rank%2 == 1) {
		for(i=0;i<N/numblocks+2;i++) {
			g1[i][N/2] = g2[i][N/2] = 75;
		}
	}
	

	print_grid(g1);

	srow = 2; frow = N/size-1;	

	up = rank-2;		
	down = rank+2;
	if(rank == 0 || rank == 1)
		up = MPI_PROC_NULL;
	if(rank == size-1 || rank = size-2 )
		down = MPI_PROC_NULL;

	
	/* Solve grids! */
	for(i=0;i<100;i++) {

		/* solve edge points */
		if(rank != 0 || rank != 1) {
			j=1;
			for(k=1;k<N-1;k++) {
				g2[j][k] = 0.25*(g1[j-1][k] + g1[j+1][k] + g1[j][k-1] + g1[j][k+1]);
			}
		}
		if (rank != size-1 || rank != size-2 ) {
			j=5;
			for(k=1;k<N-1;k++) {
				g2[j][k] = 0.25*(g1[j-1][k] + g1[j+1][k] + g1[j][k-1] + g1[j][k+1]);
			}
		}
		
		/* Start halo exchange */
		MPI_Irecv(g2[0], N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &req[0]);
		MPI_Irecv(g2[N/size+1], N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &req[2]);

		MPI_Isend(g2[N/size], N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &req[1]);
		MPI_Isend(g2[1], N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &req[3]);

		/* solve inner points */
		for(j=srow;j<=frow;j++) {
			for(k=1;k<N-1;k++) {
				g2[j][k] = 0.25*(g1[j-1][k] + g1[j+1][k] + g1[j][k-1] + g1[j][k+1]);
			}
		}

		/* Complete halo exchange */
		MPI_Waitall(4, req, stat);

		/* Swap grids */
		tmpgrid = g1;
		g1 = g2;
		g2 = tmpgrid;
		
	}
	
	print_grid(g1);
	
	MPI_Finalize();
}























