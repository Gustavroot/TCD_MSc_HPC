#include <stdio.h>
#include <mpi.h>

#define N 25

int main(int argc, char *argv[]) {

	int rank, size, err, n;
	MPI_Offset offset;
	int mydata[N];
	char errstr[1000];

	MPI_File fp;
	MPI_Status stat;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	offset = rank*25*sizeof(int);

	fp = MPI_FILE_NULL;

	err = MPI_File_open(MPI_COMM_WORLD, "ontfile", MPI_MODE_RDONLY, MPI_INFO_NULL, &fp);
	if(fp == MPI_FILE_NULL) {
		MPI_Error_string(err, errstr, &n);
		printf("Failed to open file! %s\n", errstr);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_File_seek(fp, offset, MPI_SEEK_SET);
	MPI_File_read(fp, mydata, 25, MPI_INT, &stat);
	printf("%d: mydata[11] = %d\n", rank, mydata[11]);

	MPI_File_seek(fp, 0, MPI_SEEK_SET);

	MPI_File_read_at(fp, offset, mydata, 25, MPI_INT, &stat);
	
	printf("%d: read_at: mydata[11] = %d\n", rank, mydata[11]);

	MPI_File_close(&fp);

	MPI_Finalize();


}






