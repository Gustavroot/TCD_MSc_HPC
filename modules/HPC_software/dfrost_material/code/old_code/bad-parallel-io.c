

FILE *fp;
char filename[100];

MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

sprintf(format, "output.%%%d.%dd", log10(size), 0);

sprintf(filename, format, rank);

fp = fopen(filename, "w");
