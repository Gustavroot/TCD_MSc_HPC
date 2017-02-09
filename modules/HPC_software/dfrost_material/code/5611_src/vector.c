#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

	int A[100][100];
	int B[100];
	int rank, size, i, j;
	MPI_Datatype vec;
	MPI_Status stat;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Type_vector(100, 1, 100, MPI_INT, &vec);
	MPI_Type_commit(&vec);

	if(rank == 0) {

		for(i=0;i<100;i++)
			for(j=0;j<100;j++) 
				A[i][j] = 100*i+j;

		MPI_Send(&A[0][3], 1, vec, 1, 0, MPI_COMM_WORLD);

	} else if(rank == 1) {
		// MPI_Recv(&A[0][3], 1, vec, 0, 0, MPI_COMM_WORLD, &stat);
		// printf("A[3][3] = %d\n", A[3][3]);
		MPI_Recv(B, 100, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
		printf("B[3] = %d\n", B[3]);
	}


	MPI_Finalize();
}









