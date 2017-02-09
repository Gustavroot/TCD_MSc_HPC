#include <mpi.h>
#include <stdio.h>

struct bad {

	double x;
	double z;
	int y;
	char c;
	char p;
	struct bad *next;
	double *foo;
};

int main(int argc, char *argv[]) {

	int A[100];
	int rank, size;
	MPI_Datatype two_int;
	MPI_Status stat;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Type_contiguous(2, MPI_INT, &two_int);
	MPI_Type_commit(&two_int);
	if(rank == 0) {

		A[0] = 100; A[1]=200;
		MPI_Send(A, 50, two_int, 1, 0, MPI_COMM_WORLD);
	} else {
		MPI_Recv(A, 50, two_int, 0, 0, MPI_COMM_WORLD, &stat);
		printf("A[1] = %d\n", A[1]);

	}



	MPI_Finalize();

}
