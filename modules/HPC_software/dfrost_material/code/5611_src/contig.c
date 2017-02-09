#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

	int A[100];
	int rank, size, i;
	MPI_Datatype newtype;
	MPI_Status stat;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank < 2)  {
		MPI_Type_contiguous(10, MPI_INT, &newtype);
		MPI_Type_commit(&newtype);
	} else {
		sleep(5);
		//MPI_Type_contiguous(10, MPI_INT, &newtype);
		//MPI_Type_commit(&newtype);
	}
	

	if(rank == 0) {

		for(i=0;i<100;i++)
			A[i] = 100+i;
		MPI_Send(A, 100, MPI_INT, 1, 0, MPI_COMM_WORLD);

	} else if(rank == 1) {
		MPI_Recv(A, 10, newtype, 0, 0, MPI_COMM_WORLD, &stat);
		printf("A[22] = %d\n", A[22]);
	}


	MPI_Finalize();
}









