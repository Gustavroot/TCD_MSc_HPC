#include<mpi.h>
#include <stdio.h>
#include <stdlib.h>

int N = 500;

int main(int argc, char *argv[]) {

	MPI_Status stat;
	double *a, *b;
	int rank, size, i, j;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	for(j=0;j<10000;j++) {
		a = (double *)malloc(N*sizeof(double));
		b = (double *)malloc(N*sizeof(double));

		if(a == NULL || b == NULL) {
			printf("Malloc failed on %d\n", rank);
			MPI_Abort(MPI_COMM_WORLD, 22);
		}

		for(i=0;i<N;i++) {
			a[i] = b[i] = i;
		}


		MPI_Send(a, N, MPI_DOUBLE, (rank+1)%size, 0, MPI_COMM_WORLD);
		MPI_Recv(b, N, MPI_DOUBLE, (rank-1+size)%size, 0, MPI_COMM_WORLD, &stat);

		if(rank == 0)
			printf("No deadlock %d!!\n", N);

		free(a); free(b);
		N++;
	}

	MPI_Finalize();
	return 0;
}
