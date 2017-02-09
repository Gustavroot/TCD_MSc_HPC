#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#define N 100000

int main(int argc, char *argv[]) {

	int size, rank;
	int i;
	double *a, *b;
	double *pa, *pb;
	double dp, pdp, tmp;
	MPI_Status stat;
	struct timeval start, finish;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	if(rank == 0) {
		/* Alloc and init a[] and b[] */
		a = malloc(N*sizeof(double));
		b = malloc(N*sizeof(double));

		for(i=0;i<N;i++) {
			a[i] = drand48();
			b[i] = drand48();
		}

		/* Calc dp in serial */
		gettimeofday(&start, NULL);
		dp = 0;
		for(i=0;i<N;i++) {
			dp += a[i]*b[i];
		}
		gettimeofday(&finish, NULL);
		printf("Serial dp = %f\t%f\n", dp, finish.tv_sec-start.tv_sec + 1e-6*(finish.tv_usec-start.tv_usec));

	}

	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);

	/* Alloc partial a[] and b[] */
	pa = malloc(N/size * sizeof(double));
	pb = malloc(N/size * sizeof(double));

	gettimeofday(&start, NULL);

#ifdef SEND
	/* Divvy up a[] and b[] */
	if(rank == 0) {
		/* Local copy for rank 0 pa and pb */
		for(i=0;i<N/size;i++) {
			pa[i] = a[i];
			pb[i] = b[i];
		}

		/* Send stuff to each other rank */
		for(i=1;i<size;i++) {
			MPI_Send(&a[i*N/size], N/size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			MPI_Send(&b[i*N/size], N/size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}

	} else {
		/* Recv pa[] and pb[] from rank 0 */
		MPI_Recv(pa, N/size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
		MPI_Recv(pb, N/size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);

	}
#else
	MPI_Scatter(a, N/size, MPI_DOUBLE, pa, N/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(b, N/size, MPI_DOUBLE, pb, N/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#endif
	/* Calc partial dot prod */
	pdp = 0;
	for(i=0;i<N/size;i++) {
		pdp += pa[i]*pb[i];
	}
	// printf("%d: pdp = %f\n", rank, pdp);

	/* Get results back to rank 0 */
#ifdef SEND
	if(rank == 0) {
		/* Recv from all other ranks and sum */
		for(i=1;i<size;i++) {
			MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &stat);
			pdp += tmp;
		}
		gettimeofday(&finish, NULL);
		printf("Parallel dot product = %f\t%f\n", pdp, finish.tv_sec-start.tv_sec + 1e-6*(finish.tv_usec-start.tv_usec));
	} else {
		/* Send our pdp to rank 0 */
		MPI_Send(&pdp, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
#else
	MPI_Allreduce(&pdp, &dp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	printf("%d: Parallel Parallel dot product = %f\n", rank, dp);

#endif

	for(i=0;i<100;i++) 
		MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();

	return 0;
}














