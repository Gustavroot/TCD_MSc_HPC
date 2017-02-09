#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {

	int rank, size;
	int A[10], i;
	int B[10], sleeptime, length, recvlen;
	MPI_Status stat;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank == 0) {
		for(i=0;i<size-1;i++) {
			MPI_Recv(B, 10, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
			MPI_Get_count(&stat, MPI_INT, &recvlen);

			printf("%d: Got %d of %d from %d\n",
				rank, B[0], recvlen, stat.MPI_SOURCE);
			MPI_Send(A, 10, MPI_INT, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);
		}

	} else {
		A[0] = rank;	
		srandom(rank);
		sleeptime = random()%5;
		length = 1+random()%10;
		printf("%d: sleeping for %d then sending %d items\n", rank, sleeptime, length);
		sleep(sleeptime);
		MPI_Send(A, length, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Recv(B, 10, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);
		printf("%d: got stuff from 0\n", rank);
	}


	MPI_Finalize();
}
