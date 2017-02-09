#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100

int main(int argc, char *argv[]) {
	
	int size, rank, insize, *in, *in2, offset;
	MPI_File fp;
	MPI_Status stat;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	insize = N/size;
	in = malloc(insize * sizeof(int));
	in2 = malloc(insize * sizeof(int));
	offset = rank * insize * sizeof(int);

	MPI_File_open(MPI_COMM_WORLD, "intfile", MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);

	MPI_File_read_at(fp, offset, in, insize, MPI_INT, &stat);

	MPI_File_seek(fp, offset, MPI_SEEK_SET);
	MPI_File_read(fp, in2, insize, MPI_INT, &stat);
	MPI_File_close(&fp);

	printf("%d: %d %d\n", rank, in[0], in2[0]);

	free(in);
	MPI_Finalize();
	return 0;

}
