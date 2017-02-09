#include <unistd.h>
#include <stdio.h>
#include <mpi.h>

#define N 100000
int main(int argc, char *argv[]) {

	int rank, size;
	char host[100];
	int foo[N], foo2[N];
	double x[N];
	MPI_Status stat;

	MPI_Init(&argc, &argv);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Barrier(MPI_COMM_WORLD);
	gethostname(host, 100);

	if(rank %2 == 0) {
		printf("%d: I'm an even process\n", rank);
	} else {
		printf("%d: I'm an odd process\n", rank);
	}

	if(rank == 0) {
		foo[0] = random();	
		foo[1] = random();	
		printf("%d: Foo[0] = %d (before send)\n", rank, foo[0]);
		printf("%d: Foo[1] = %d (before send)\n", rank, foo[1]);
		MPI_Send(foo, 100, MPI_INT, 1, 0, MPI_COMM_WORLD);
	} else if(rank == 1) {
		MPI_Recv(x, 100, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);	
		printf("%d: Foo[0] = %e (after recv)\n", rank, x[0]);
	

	}

	MPI_Barrier(MPI_COMM_WORLD);

	printf("%d: About to send\n", rank);
	MPI_Send(foo, N, MPI_INT, (rank+1)%size, 0, MPI_COMM_WORLD);
	printf("%d: About to recv\n", rank);
	MPI_Recv(foo2, N, MPI_INT, (size+rank-1)%size, 0, MPI_COMM_WORLD, &stat);
	
	MPI_Barrier(MPI_COMM_WORLD);
	// printf("Hello world from %d of %d on %s\n", rank, size, host);
	MPI_Barrier(MPI_COMM_WORLD);

	
	printf("%d: Waiting for finalize\n", rank);

	MPI_Finalize();

	return 0;
}
