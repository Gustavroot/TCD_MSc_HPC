#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>

int N=10000000;

int foo() {

	printf("Executing foo\n");
	N=N/2;
	return random()%2;

}


int main() {

	int *A;
	int i, max;
	double t;
	struct timeval start, finish;

	A = malloc(N * sizeof(int));

	for(i=0;i<N;i++) {
		A[i] = random();
	}

	max = INT_MIN;

	gettimeofday(&start, NULL);
#pragma omp parallel for if(foo() == 0) lastprivate(i) schedule(runtime)
	for(i=0;i<N;i++) {
		if(A[i] > max)
#pragma omp critical
		{
		if(A[i] > max)
			max = A[i];
		}
	}
	gettimeofday(&finish, NULL);

	t = (finish.tv_sec-start.tv_sec) + 1e-6*(finish.tv_usec-start.tv_usec);	
	printf("Max value = %d (%f) %d iterations\n", max, t, i);

}
