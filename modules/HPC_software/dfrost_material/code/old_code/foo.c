#include <omp.h>
#include <stdio.h>
#define N 100

int x=N;

int main() {


#pragma omp parallel num_threads(8)
{

	int id;
	id = omp_get_thread_num();
	printf("%d: Hello world\n", id);
}
	
#pragma omp parallel
{

	int id;
	id = omp_get_thread_num();
	printf("%d: Hello again world\n", id);
}

	return 0;
}
