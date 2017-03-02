//=============================================================================================
// Name        		: mpiHelloWorld.cu
// Author      		: Jose Refojo
// Version     		:
// Creation date	:	14-11-10
//=============================================================================================

#define MATRIX_SIZE_N 40
#define MATRIX_SIZE_M 20

#include "stdio.h"
#include "time.h"
#include "mpi.h"

#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>

int main(int argc, char *argv[]) {
	int         my_rank;       // rank of process
	int         p;             // number of processes
	int         source;        // rank of sender
	int         dest;          // rank of receiver
	int         tag = 0;       // tag for messages
	char        message[100];  // storage for message
	MPI_Status  status;        // return status for receive

	// Start up MPI
	MPI_Init(&argc, &argv);

	// Find out process rank
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	// Find out number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	printf("This is process %d \n", my_rank);

	if (my_rank != 0) {
		// Display the processor that's working
		// Create message
		sprintf(message, "Greetings from process %d!",my_rank);
		dest = 0;
		// Use strlen+1 so that '\0' gets transmitted
		MPI_Send(message, strlen(message)+1, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
	} else { // my_rank == 0
		for (source = 1; source < p; source++) {
			MPI_Recv(message, 100, MPI_CHAR, source, tag,MPI_COMM_WORLD, &status);
			printf("process %d received '%s'\n", my_rank,message);
		}
	}

	// Shut down MPI
	MPI_Finalize();
}
