#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#define N 10000000

int main(int argc, char *argv[]) {

	MPI_Status stat;
	double *x, *y, dp;
	double *px, *py, pdp;
	int size, rank;
	int i;
	struct timeval start, finish;


	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank == 0) {
		x = malloc(N * sizeof(double));
		y = malloc(N * sizeof(double));

		/* Initialize vectors */
		for(i=0;i<N;i++) {
			x[i] = drand48();
			y[i] = drand48();
		}

		gettimeofday(&start, NULL);
		/* Calculate DP in serial */
		dp = 0;
		for(i=0;i<N;i++)
			dp += x[i]*y[i];	
		gettimeofday(&finish, NULL);

		printf("Serial dp = %f in %f\n", dp, (finish.tv_sec-start.tv_sec) + (finish.tv_usec-start.tv_usec)*1e-6);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/* Allocate sub vectors on each task */
	px = malloc(N/size * sizeof(double));
	py = malloc(N/size * sizeof(double));

	gettimeofday(&start, NULL);
	/* Distribute x and y to each task */
#ifdef SENDRECV
	if(rank == 0) {
		/* Copy locally into my px and py */
		for(i=0;i<N/size;i++) {
			px[i] = x[i];
			py[i] = y[i];
		}
		/* Send rest of x and y to other tasks */
		for(i=1;i<size;i++) {
			MPI_Send(&x[i*N/size], N/size, MPI_DOUBLE ,i, 9, MPI_COMM_WORLD);
			MPI_Send(&y[i*N/size], N/size, MPI_DOUBLE ,i, 9, MPI_COMM_WORLD);
		}

	} else {
		/* Get stuff from rank 0 */
		MPI_Recv(px, N/size, MPI_DOUBLE, 0, 9, MPI_COMM_WORLD, &stat);
		MPI_Recv(py, N/size, MPI_DOUBLE, 0, 9, MPI_COMM_WORLD, &stat);
	}
#endif

#ifdef SCATTER

	MPI_Scatter(x, N/size, MPI_DOUBLE, px, N/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(y, N/size, MPI_DOUBLE, py, N/size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

#endif

	
	/* Calculate partial DP on each task */
	pdp = 0;
	for(i=0;i<N/size;i++) {
		pdp += px[i]*py[i];
	}	

#ifdef SENDRECV
	/* Sum up partials on rank 0 and print answer */
	if(rank == 0) {
		dp = pdp;
		for(i=1;i<size;i++) {
			MPI_Recv(&pdp, 1, MPI_DOUBLE,  i, 33, MPI_COMM_WORLD, &stat);
			dp += pdp;
		}
		
		gettimeofday(&finish, NULL);
		printf("Parallel dp = %f in %f\n", dp, (finish.tv_sec-start.tv_sec) + (finish.tv_usec-start.tv_usec)*1e-6);

	} else {

		MPI_Send(&pdp, 1, MPI_DOUBLE, 0, 33, MPI_COMM_WORLD);
	}
#endif

	MPI_Allreduce(&pdp, &dp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	gettimeofday(&finish, NULL);
	printf("%d: Parallel dp = %f in %f\n", rank, dp, (finish.tv_sec-start.tv_sec) + (finish.tv_usec-start.tv_usec)*1e-6);




	MPI_Finalize();
}




















