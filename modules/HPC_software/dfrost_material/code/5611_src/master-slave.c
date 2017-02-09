#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void do_work(int x) {
	printf("Slave %d starting job\n", x);
	sleep(5);
}

int main(int argc, char *argv[]) {

	int size, rank;
	int src;
	int i;
	int A[10];
	MPI_Status stat;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	if(rank == 0) {
		/* I am the master */
		for(i=1;i<size;i++) {
			printf("MASTER: sending job to %d\n", i);
			MPI_Send(A, 10, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		for(i=1;i<100;i++) {
			MPI_Recv(A, 10, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
			src = stat.MPI_SOURCE;
			printf("MASTER: sending job to %d\n", src);
			MPI_Send(A, 10, MPI_INT, src, 0, MPI_COMM_WORLD);
		}
		
	} else {
		/* I am a slave */
		while(1) {
			MPI_Recv(A, 10, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);

			do_work(rank);

			MPI_Send(A, 10, MPI_INT, 0, 0, MPI_COMM_WORLD);

		}
	}


	printf("Hello world %d of %d\n", rank, size);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
