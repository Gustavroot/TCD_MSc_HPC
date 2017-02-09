#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000

int main() {

	int num_cpus;

#ifdef _OPENMP
	num_cpus = omp_get_num_procs();
#endif

	printf("There are %d cores in this machine\n", num_cpus);

#pragma omp parallel
	{
		int thread_id;
#ifdef _OPENMP
		thread_id = omp_get_thread_num();
#endif
		printf("%d: Pre Hello world\n", thread_id);
	}

	printf("-----------------------------------------------\n");

#ifdef _OPENMP
	omp_set_num_threads(5);
#endif

#pragma omp parallel
	{
		int thread_id;
#ifdef _OPENMP
		thread_id = omp_get_thread_num();
#endif
		printf("%d: Hello world\n", thread_id);
	}


	printf("-----------------------------------------------\n");
#pragma omp parallel num_threads(7)
	{
		int thread_id;
#ifdef _OPENMP
		thread_id = omp_get_thread_num();
#endif
		printf("%d: Hello again\n", thread_id);
	}
	
	return 0;
}
