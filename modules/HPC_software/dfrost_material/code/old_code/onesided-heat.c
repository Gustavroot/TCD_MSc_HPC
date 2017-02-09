#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>


int setup();
int solve();
void print_grid(double **g);

double *g1_data, *g2_data, **g1, **g2;
int edges[4] = {100, 50, 25, 0};
int rank, size;
int rows, cols, lrows;

int main(int argc, char *argv[]) {


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	cols = 12+2; rows = 24;
	if(rows%size != 0) {
		if(rank == 0)
			printf("Rows (%d) must be divisible by nprocs (%d)\n", rows, size);
		MPI_Finalize();
		return 1;
	}
	lrows = rows/size + 2; // Space for halo points

	setup();

	solve();
	

	MPI_Finalize();
	return 0;
}

int solve() {
	MPI_Win win1, win2, *active;
	MPI_Status stat;
	MPI_Group wgrp, grp;
	int iteration;
	int i, j, k;
	int nup, ndown;
	double **tmp, *tmp_data;

	MPI_Win_create(g1_data, lrows*cols*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
	MPI_Win_create(g2_data, lrows*cols*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win2);

	active = &win1;
	MPI_Win_fence(0, *active);

	/* Figure out who my neighbours are */
	nup = rank-1;
	ndown = rank+1;
	if(rank == 0)
		nup = MPI_PROC_NULL;
	if(rank == size-1)
		ndown = MPI_PROC_NULL;

#if defined PSCW
	MPI_Win_get_group(*active, &wgrp);
	if(rank == 0) {
		int neighbours[1];
		neighbours[0] = ndown;
		MPI_Group_incl(wgrp, 1, neighbours, &grp);
	} else if(rank == size-1) {
		int neighbours[1];
		neighbours[0] = nup;
		MPI_Group_incl(wgrp, 1, neighbours, &grp);
	} else {
		int neighbours[2];
		MPI_Win_get_group(*active, &wgrp);
		neighbours[0] = nup;
		neighbours[1] = ndown;
		MPI_Group_incl(wgrp, 2, neighbours, &grp);
	}
	MPI_Group_free(&wgrp);
#endif
			
	for(iteration=0;iteration<100;iteration++) {
		/* Halo exchange */

#if defined SEND
		// printf("%d: sending to %d\n", rank, nup);
		MPI_Send(&g1[1][0], cols, MPI_DOUBLE, nup, 0, MPI_COMM_WORLD);
		// printf("%d: receiving from %d\n", rank, ndown);
		MPI_Recv(&g1[lrows-1][0], cols, MPI_DOUBLE, ndown, 0, MPI_COMM_WORLD, &stat);

		// printf("%d: sending to %d\n", rank, ndown);
		MPI_Send(&g1[lrows-2][0], cols, MPI_DOUBLE, ndown, 0, MPI_COMM_WORLD);
		// printf("%d: receiving from %d\n", rank, nup);
		MPI_Recv(&g1[0][0], cols, MPI_DOUBLE, nup, 0, MPI_COMM_WORLD, &stat);

#elif defined SENDRECV
		MPI_Sendrecv(&g1[1][0], cols, MPI_DOUBLE, nup, 0,&g1[lrows-1][0], cols, MPI_DOUBLE, ndown, 0, MPI_COMM_WORLD, &stat);
		MPI_Sendrecv(&g1[lrows-2][0], cols, MPI_DOUBLE, ndown, 0, &g1[0][0], cols, MPI_DOUBLE, nup, 0, MPI_COMM_WORLD, &stat);

#elif defined FENCE
		MPI_Win_fence(MPI_MODE_NOSTORE & MPI_MODE_NOPUT, *active);

		MPI_Get(&g1[lrows-1][0], cols, MPI_DOUBLE, ndown, cols, cols, MPI_DOUBLE, *active);
		MPI_Get(&g1[0][0], cols, MPI_DOUBLE, nup, (lrows-2)*cols, cols, MPI_DOUBLE, *active);
		
		MPI_Win_fence(MPI_MODE_NOSUCCEED, *active);

#elif defined LOCK
		if(ndown != MPI_PROC_NULL) {
			MPI_Win_lock(MPI_LOCK_SHARED, ndown, 0, *active);
			MPI_Get(&g1[lrows-1][0], cols, MPI_DOUBLE, ndown, cols, cols, MPI_DOUBLE, *active);
			MPI_Win_unlock(ndown, *active);
		}

		if(nup != MPI_PROC_NULL) {
			MPI_Win_lock(MPI_LOCK_SHARED, nup, 0, *active);
			MPI_Get(&g1[0][0], cols, MPI_DOUBLE, nup, (lrows-2)*cols, cols, MPI_DOUBLE, *active);
			MPI_Win_unlock(nup, *active);
		}
		MPI_Barrier(MPI_COMM_WORLD); // Required as there is no global co-ordination as with fencing

#elif defined PSCW
		MPI_Win_post(grp, 0, *active);
		MPI_Win_start(grp, 0, *active);

		MPI_Get(&g1[lrows-1][0], cols, MPI_DOUBLE, ndown, cols, cols, MPI_DOUBLE, *active);
		MPI_Get(&g1[0][0], cols, MPI_DOUBLE, nup, (lrows-2)*cols, cols, MPI_DOUBLE, *active);

		MPI_Win_complete(*active);
		MPI_Win_wait(*active);

		MPI_Barrier(MPI_COMM_WORLD);


#else
		if(rank == 0)
			printf("No exchange method chosen at compile time! Exiting\n");
		MPI_Abort(MPI_COMM_WORLD, 1);

#endif

		/* Update */
		for(i=1;i<lrows-1;i++) {
			for(j=1;j<cols-1;j++) {
				g2[i][j] = (g1[i-1][j] + g1[i+1][j] + g1[i][j-1] + g1[i][j+1])*0.25;
			}
		}
		
		/* Swap */
		if(active == &win1) {
			active = &win2;
		} else {
			active = &win1;
		}
		tmp=g1; g1=g2; g2=tmp;
#ifdef LOCK
		MPI_Barrier(MPI_COMM_WORLD);
#endif

	}

	print_grid(g1);

	MPI_Win_free(&win1);
	MPI_Win_free(&win2);
	return 0;
}

int setup() {

	int i;

	/* Allocate the two grids */
	g1_data = malloc(lrows*cols*sizeof(double));
	g2_data = malloc(lrows*cols*sizeof(double));
	memset(g1_data, 0, lrows*cols*sizeof(double));
	memset(g2_data, 0, lrows*cols*sizeof(double));

	g1 = malloc(lrows*sizeof(double *));
	g2 = malloc(lrows*sizeof(double *));
	for(i=0;i<lrows;i++) {
		g1[i] = &g1_data[cols*i];
		g2[i] = &g2_data[cols*i];
	}

	/* Set initial values */
	for(i=0;i<lrows;i++) {
		g1[i][0] = g2[i][0] = edges[0];
		g1[i][cols-1] = g2[i][cols-1] = edges[1];
	}
	if(rank == 0) {
		for(i=0;i<cols;i++)
			g1[0][i] = g2[0][i] = edges[2];
	} else if(rank == size-1) {
		for(i=0;i<cols;i++)
			g1[lrows-1][i] = g2[lrows-1][i] = edges[3];
	}
	
	return 0;
}

void print_grid(double **g) {
	int i, j, k;
	int srow, erow;
	
	srow=1; erow=lrows-1;
	if(rank == 0) srow = 0;
	if(rank == size-1) erow=lrows;

	for(i=0;i<size;i++) {

		if(rank == i) {
			for(j=srow;j<erow;j++) {
				for(k=0;k<cols;k++)
					printf("%2.2f\t", g[j][k]);
				printf("\n");
			}
			fflush(stdout);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}
