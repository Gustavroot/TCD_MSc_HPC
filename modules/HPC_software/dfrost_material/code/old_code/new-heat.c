#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>

#define N 20
int rank, size;

void print_grid(double **X) {

	int i, j, k;
	if(rank ==0 ) {
		printf("--------------------------------------------------------\n");
	}

	MPI_Gather(..., 3, MPI_COMM_WORLD);
	if(rank == 3) {
		printf("....");

	}
	for(k=0;k<size;k++) {
		if(rank == k) {

			for(i=1;i<=N/size;i++) {
				printf("%d: ", rank);
				for(j=0;j<N;j++) {
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
	MPI_Status stat;
	

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	/* Allocate space for two grids */
	g1 = malloc( (N/size + 2) * sizeof(double *));
	tmp = malloc( (N/size + 2) * N * sizeof(double));
	for(i=0;i<N/size+2;i++) {
		g1[i] = &tmp[N * i];
	}
	
	g2 = malloc( (N/size + 2) * sizeof(double *));
	tmp = malloc( (N/size + 2) * N * sizeof(double));
	for(i=0;i<N/size+2;i++) {
		g2[i] = &tmp[N * i];
	}
	
	/* Initialize grids */	
	/* Top row set to 100 */
	if(rank == 0) {
		for(i=0;i<N;i++) {
			g1[1][i] = g2[1][i] = 100;
		}
	}
	/* Bottom row set to 200 */
	if(rank == size-1) {
		for(i=0;i<N;i++) {
			g1[N/size][i] = g2[N/size][i] = 200;
		}
	}
	/* Set both size to 50 */
	for(i=0;i<N/size+2;i++) {
		g1[i][0] = g1[i][N-1] = 50;
		g2[i][0] = g2[i][N-1] = 50;
	}
	

	print_grid(g1);

	srow = 1; frow = N/size;	
	if(rank==0)
		srow = 2;
	if(rank == size-1)
		frow = N/size-1;

	up = rank-1;		
	down = rank+1;
	if(rank == 0)
		up = MPI_PROC_NULL;
	if(rank == size-1)
		down = MPI_PROC_NULL;

	
	/* Solve grids! */
	for(i=0;i<100;i++) {

#ifdef CASCADE
		/* Exchange halo rows */
		if(rank == 0) {
			/* Top */
			MPI_Send(g1[N/size], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
			MPI_Recv(g1[N/size+1], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &stat);
		} else if(rank == size-1) {
			/* Bottom */
			MPI_Recv(g1[0], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(g1[1], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
		} else {
			/* Everyone else */
			MPI_Recv(g1[0], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(g1[N/size], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
			MPI_Recv(g1[N/size+1], N, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(g1[1], N, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
		}
#endif
		
#ifdef CHECK

		if(rank%2 == 0) {
			MPI_Recv(g1[0], N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(g1[N/size], N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD);
			MPI_Recv(g1[N/size+1], N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(g1[1], N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);
		} else {
			MPI_Send(g1[N/size], N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD);
			MPI_Recv(g1[0], N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(g1[1], N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);
			MPI_Recv(g1[N/size+1], N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &stat);
		}

	x = stat.MPI_SOURCE;
#endif

#ifdef SENDRECV

	/* Joint send/recv downwards */
	MPI_Sendrecv(g1[N/size], N, MPI_DOUBLE, down, 0,g1[0], N, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &stat); 
	/* Joint send/recv upwards */
	MPI_Sendrecv(g1[1], N, MPI_DOUBLE, up, 0, g1[N/size+1], N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &stat);

#endif

		/* Update g2 with values from g1 */
		for(j=srow;j<=frow;j++) {
			for(k=1;k<N-1;k++) {
				g2[j][k] = 0.25*(g1[j-1][k] + g1[j+1][k] + g1[j][k-1] + g1[j][k+1]);
			}
		}

		/* Swap grids */
		tmpgrid = g1;
		g1 = g2;
		g2 = tmpgrid;
		
	}
	
	print_grid(g1);
	
	MPI_Finalize();
}























