#include <stdio.h>
#include <omp.h>
#define N 100

int main() {

	int id, var;
	int i;

	var = 501;

#pragma omp parallel for private(id) firstprivate(var) lastprivate(var)
	for(i=0;i<N;i++) {
		id = omp_get_thread_num();
		printf("%d: var = %d\n", id, var);
		var = i;
	}

	printf("Var = %d\n", var);

}
