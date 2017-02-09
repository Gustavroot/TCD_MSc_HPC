#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {

	int size, rank;

	MPI_Init(&argc, &argv);
	
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	MPI_Barrier(MPI_COMM_WORLD);
	printf("Hello World from %d of %d\n", rank, size);
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
}
