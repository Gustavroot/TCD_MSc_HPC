#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 4

int main(int argc, char *argv[]) {
	
	MPI_File fp;
	MPI_Status stat;
	MPI_Datatype newtype;
	int **A;
	int err;
	char errstr[1024];
	int *A_data;
	int rank, size;
	int offset;
#ifdef DEBUG
	int debugrank = 1;
#endif
	int i, j, n;
	int fullsize[2] = {2*N, 2*N};
	int localsize[2] = {N, N};
	int starts[2] = {0, 0};

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	
	MPI_Comm_size(MPI_COMM_WORLD, &size);	

	A_data = malloc(N * N * sizeof(int));
	A = malloc(N*sizeof(int *));
	for(i=0;i<N;i++) {
		A[i] = &A_data[i*N];
	}


	for(i=0;i<N;i++) {
		for(j=0;j<N;j++) {
			if(rank < 2) {
				A[i][j] = rank*N + i*8 + j;
			} else {
				A[i][j] = 2*N*N + (rank-2)*N + i*8 + j;
			}
#ifdef DEBUG
			if(rank == debugrank)
				printf("%d\t", A[i][j]);
#endif
		}
#ifdef DEBUG
		if(rank == debugrank)
			printf("\n");
#endif
	}

	MPI_Type_create_subarray(2, fullsize, localsize, starts, MPI_ORDER_C, MPI_INT, &newtype);
	MPI_Type_commit(&newtype);

	switch(rank) {
		case 0: offset = 0; break;
		case 1: offset = N*sizeof(int); break;
		case 2: offset = (2*N*N)*sizeof(int); break;
		case 3: offset = (N+2*N*N)*sizeof(int); break;
	}

	err = MPI_File_open(MPI_COMM_WORLD, "matrix_file", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fp);
	
        if(fp == MPI_FILE_NULL) {
                MPI_Error_string(err, errstr, &n);
                printf("Failed to open file! %s\n", errstr);
                MPI_Abort(MPI_COMM_WORLD, 1);
        }

	MPI_File_set_view(fp, offset, MPI_INT, newtype, "native", MPI_INFO_NULL);
	MPI_File_write_all(fp, &(A[0][0]), N*N, MPI_INT, &stat);

	MPI_File_close(&fp);
	

	MPI_Finalize();

}
