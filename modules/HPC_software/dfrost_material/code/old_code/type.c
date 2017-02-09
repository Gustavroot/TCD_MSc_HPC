#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

	MPI_Datatype row;
	MPI_Status stat;

	void *p;
	int pos;

	int A[10][10];
	int i, j;
	int x, y;
	double z;
	char c; 

	int rank, size;
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	p = malloc(10000);

	MPI_Type_contiguous(10, MPI_DOUBLE, &row);
	MPI_Type_commit(&row);

	if(rank == 0) {
		for(i=0;i<10;i++) {
			A[0][i] = random()%20;
		}	

		x = 4; y = 11; z = -3.1415; c = 'p';
		pos = 0;
		MPI_Pack(&x, 1, MPI_INT, p, 10000, &pos, MPI_COMM_WORLD);
		MPI_Pack(&y, 1, MPI_INT, p, 10000, &pos, MPI_COMM_WORLD);
		MPI_Pack(&z, 1, MPI_DOUBLE, p, 10000, &pos, MPI_COMM_WORLD);

		MPI_Send(p, 10000, MPI_PACKED, 1, 77, MPI_COMM_WORLD);
		MPI_Send(A, 1, row, 1, 0, MPI_COMM_WORLD);

	} else if (rank == 1) {

		MPI_Recv()
		MPI_Unpack( ...)
		MPI_Unpack( ...)
		MPI_Unpack( ...)
		
		printf("Before: A[0][0] = %d\n", A[0][0]);
		MPI_Recv(A, 1, row, 0, 0, MPI_COMM_WORLD, &stat);
		// MPI_Recv(A, 10, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
		printf("After: A[0][0] = %d\n", A[0][0]);
	}


	MPI_Type_free(&row);

	MPI_Finalize();
}











