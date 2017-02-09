#include <unistd.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>


int main(int argc, char *argv[]) {

	int rank, size;
	int i;
	int A[100], B[100];
	char hostname[100];
	MPI_Status stat;

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	gethostname(hostname, 100);

	if(rank == 2) {
		// sleep(3);
	}
	
	if(rank == 0) {
		for(i=0;i<100;i++)
			A[i] = random();
	}

	printf("%d: About to enter barrier\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
	printf("Hello world from %d of %d on %s\n", rank, size, hostname);

	if(rank == 0) {
		MPI_Send(A, 100, MPI_INT, 1, 0, MPI_COMM_WORLD);
	} else if (rank == 1) {
		printf("%d: Before recv B[2] = %d\n", rank, B[2]);
		MPI_Recv(B, 100, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
		printf("%d: After recv B[2] = %d\n", rank, B[2]);

		
	}
	
	MPI_Finalize();

	return 0;
}














