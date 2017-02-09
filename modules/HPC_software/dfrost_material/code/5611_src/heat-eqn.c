#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define M 20
#define N 10

int rank, size;
int lcols, lrows;

void print_grid(double **g) {

	int k, i, j;
	
	for(k=0;k<size;k++) {

		if(rank == k) {
			for(i=1;i<lrows-1;i++) {
				printf("%d: ", rank);
				for(j=0;j<lcols;j++) {
					printf("%2.0f\t", g[i][j]);
				}	
				printf("\n");
			}
			fflush(stdout);
			usleep(1000);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	
}

int main(int argc, char *argv[]) {

	double **g1, **g2, **tmpg;
	double *p;
	int i, j, k;
	int start_row, end_row;
	int up, down;
	MPI_Status stat;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* Assume size divides M equally */
	lcols = N;
	lrows = (M/size + 2);
	
	/* Allocate our two local grid */
	g1 = malloc(lrows * sizeof(double *));
	p = malloc(lcols * lrows * sizeof(double));
	for(i=0;i<lrows;i++) {
		g1[i] = &p[i*lcols];		
	}

	g2 = malloc(lrows * sizeof(double *));
	p = malloc(lcols * lrows * sizeof(double));
	for(i=0;i<lrows;i++) {
		g2[i] = &p[i*lcols];		
	}

	up = rank-1;
	down = rank+1;
	if(rank == 0)
		up = MPI_PROC_NULL;
	if(rank == size-1)
		down = MPI_PROC_NULL;

	/* Initalize data values */
	/* Top = 100, Bottom = 50, Left = 75, Right = 25 */
	/* Left and right */
	for(i=0;i<lrows;i++) {
		g1[i][0] = g2[i][0] = 75;
		g1[i][lcols-1] = g2[i][lcols-1] = 25;
	}
	/* Top only on rank zero */
	if(rank == 0) {
		for(i=0;i<lcols;i++) {
			g1[1][i] = g2[1][i] = 100;
		}
	}
	/* Bottom only on rank size-1 */
	if(rank == size-1) {
		for(i=0;i<lcols;i++) {
			g1[lrows-2][i] = g2[lrows-2][i] = 50;
		}
	}
	
	print_grid(g1);		

	/* Start to iterate the Jacobi method */
	for(i=0;i<100;i++) {

		/* Halo exchange of rows */
		/*
		MPI_Send(row[1] to rank-1)
		MPI_Send(row[lrows-2] to rank+1)

		MPI_Recv(row[0] from rank-1)
		MPI_Recv(row[lrows-1] from rank+1) 
		*/

#ifdef CASCADE
		if(rank == 0) {
			/* Top rank only has to exchange down */
			MPI_Send(&g1[lrows-2][0], lcols, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);

			MPI_Recv(&g1[lrows-1][0], lcols, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &stat);

		} else if(rank == size-1) {
			/* Bottom rank only up */
			MPI_Recv(&g1[0][0], lcols, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &stat);

			MPI_Send(&g1[1][0], lcols, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
		} else {
			/* Everyone else both directions */
			MPI_Send(&g1[lrows-2][0], lcols, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
			MPI_Recv(&g1[0][0], lcols, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &stat);

			MPI_Send(&g1[1][0], lcols, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
			MPI_Recv(&g1[lrows-1][0], lcols, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &stat);
		}
#endif
	
#ifdef BW
		
		if(rank%2 == 0) {
			MPI_Send(&g1[lrows-2][0], lcols, MPI_DOUBLE, down, 0, MPI_COMM_WORLD);
			MPI_Send(&g1[1][0], lcols, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);

			MPI_Recv(&g1[0][0], lcols, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&g1[lrows-1][0], lcols, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &stat);
		} else {
			MPI_Recv(&g1[0][0], lcols, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&g1[lrows-1][0], lcols, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &stat);

			MPI_Send(&g1[lrows-2][0], lcols, MPI_DOUBLE, down, 0, MPI_COMM_WORLD);
			MPI_Send(&g1[1][0], lcols, MPI_DOUBLE, up, 0, MPI_COMM_WORLD);

		}

#endif

#ifdef SENDRECV
		/* Down */
		MPI_Sendrecv(&g1[lrows-2][0], lcols, MPI_DOUBLE, down, 0, 
						&g2[0][0], lcols, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &stat);
		/* Up */
		MPI_Sendrecv(&g1[1][0], lcols, MPI_DOUBLE, up, 0,
						&g1[lrows-1][0], lcols, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &stat);

#endif

		start_row = 1; end_row = lrows-1;
		if(rank == 0) {
			start_row = 2;
		} else if(rank == size-1) {
			end_row = lrows-2;
		}
		/* Update g2 from g1 */
		for(j=start_row;j<end_row;j++) {
			for(k=1;k<lcols-1;k++) {
				g2[j][k] = 0.25 * (g1[j-1][k] + g1[j+1][k] + g1[j][k-1] + g1[j][k+1]);
			}
		}

		/* Swap g1 and g2 */		
		tmpg = g1;
		g1 = g2;
		g2 = tmpg;

	}

	print_grid(g1);		
	MPI_Finalize();
	return 0;
}










