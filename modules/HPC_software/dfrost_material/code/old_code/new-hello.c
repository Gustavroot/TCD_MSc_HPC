#include <stdio.h>	/* printf etc */
#include <stdlib.h>	/* malloc/free */
#include <mpi.h>	/* MPI stuff */
#include <unistd.h>	/* gethostname() */
#include <sys/time.h>	/* gettimeofday() */


int main(int argc, char *argv[]) {

	int i, j, k;
	double *x, elapsed;
	int rank, size;
	char hostname[100];
	struct timeval t1, t2;


	MPI_Init(&argc, &argv);
	gettimeofday(&t1, NULL);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	gethostname(hostname, 100);

	printf("Hello world from %d of %d on %s\n", rank, size, hostname);

	if(rank == 0) {
		printf("%d: Separate path for rank 0\n", rank);
		MPI_Barrier(MPI_COMM_WORLD);
	} else {
		printf("%d: everyone else over here\n", rank);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	fflush(stdout);
	sleep(1);

	for(i=0;i<size;i++) {
		if(rank == i) {
			printf("In order hello from %d\n", rank);
			fflush(stdout);
			sleep(1);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	gettimeofday(&t2, NULL);
	elapsed = (t2.tv_sec-t1.tv_sec) + (t2.tv_usec-t1.tv_usec)*1e-6;
	printf("%d took %f\n", rank, elapsed);

	MPI_Finalize();
	return 0;

}
