#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DISP (100*sizeof(int))
#define N 250

int main(int argc, char *argv[]) {
	
	int buf[N];
	int size, rank, ext;
	int i, j;
	MPI_Datatype ltype, ftype;
	MPI_File fp;
	MPI_Status stat;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_File_open(MPI_COMM_WORLD, "intfile", MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);

	MPI_Type_contiguous(10, MPI_INT, &ltype);
	ext = 40 * sizeof(int);
	MPI_Type_create_resized(ltype, 0, ext, &ftype);
	MPI_Type_commit(&ftype);
	
	MPI_File_set_view(fp, DISP + rank*10*sizeof(int), MPI_INT, ftype, "native", MPI_INFO_NULL);

	// MPI_File_read(fp, buf, N, MPI_INT, &stat);
	MPI_File_read_all(fp, buf, N, MPI_INT, &stat);
	
	MPI_File_close(&fp);

	if(rank==1) {
		for(i=0;i<25;i++) {
			for(j=0;j<10;j++)
				printf("%d\t", buf[i*10+j]);
			printf("\n");
		}
	}

	MPI_Finalize();
	return 0;

}
