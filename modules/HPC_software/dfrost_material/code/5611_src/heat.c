#include <stdio.h>
#include <stdlib.h>

#define N 20

void print_grid(double **g) {
	int i, j;
	for(i=0;i<N;i++) {
		for(j=0;j<N;j++) {
			printf("%2.2f\t", g[i][j]);
		}
		printf("\n");
	}
}

int main(int argc, char *argv[]) {
	
	double **A, **B, **t;
	double *tmp;
	int i, j;
	int itcount, itlimit;

	/* Allocate 2 grids */
	A = malloc(N*sizeof(double *));
	tmp = malloc(N*N*sizeof(double));
	for(i=0;i<N;i++)
		A[i] = &tmp[i*N];
	
	B = malloc(N*sizeof(double *));
	tmp = malloc(N*N*sizeof(double));
	for(i=0;i<N;i++)
		B[i] = &tmp[i*N];
	

	/* Initialize Boundaries */
	for(i=0;i<N;i++) {
		/* Top and bottom */
		A[0][i] = A[N-1][i] = 100;
		B[0][i] = B[N-1][i] = 100;
	
		/* Left and right */
		A[i][0] = A[i][N-1] = 0;
		B[i][0] = B[i][N-1] = 0;
	}

	print_grid(A);
	itlimit = 1000;
	for(itcount=0; itcount<itlimit; itcount++) {
		/* Update B from values in A */
		for(i=1;i<N-1;i++)
			for(j=1;j<N-1;j++)
				B[i][j] = 0.25 * (A[i-1][j] + A[i+1][j] +
						A[i][j-1] + A[i][j+1] );


		/* Swap A and B */
		t = A;
		A = B;
		B = t;
	}

	printf("------------------------------------------------------------\n");
	print_grid(A);
	
}
