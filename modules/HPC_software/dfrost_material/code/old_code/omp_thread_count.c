#include <stdio.h>
#include <omp.h>

int main() {


	int n;

	n = omp_get_num_procs();

	printf("System says %d procs\n", n);

	omp_set_num_threads(7);

#pragma omp parallel num_threads(5)
	printf("%d: Hello world\n", omp_get_thread_num());

}
