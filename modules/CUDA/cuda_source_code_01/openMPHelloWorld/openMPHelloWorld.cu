//=============================================================================================
// Name        		: openMPHelloWorld.cu
// Author      		: Jose Refojo
// Version     		:
// Creation date	:	20-11-10
//=============================================================================================

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
 
int main (int argc, char *argv[]) {
	int threadId, numberOfThreads;
#pragma omp parallel private(threadId)
	{
		threadId = omp_get_thread_num();
		printf("Hello World from thread %d\n", threadId);
		#pragma omp barrier
		if ( threadId == 0 ) {
			numberOfThreads = omp_get_num_threads();
			printf("[From process: %d] There are %d threads\n",threadId,numberOfThreads);
		}
	}
  return EXIT_SUCCESS;
}

