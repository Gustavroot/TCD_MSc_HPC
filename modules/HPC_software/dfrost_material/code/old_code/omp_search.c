#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include <sys/time.h>


#define N 10000000

int main() {

	long *a;
	int max, i;
	double t;
	struct timeval t1, t2;

	srandom(0);
	a = malloc(N*sizeof(long));

	for(i=0;i<N;i++) {
		a[i] = (long)random()%(N*100);
		a[i] = i;
	}	

	max = INT_MIN;


	gettimeofday(&t1, NULL);
#pragma omp parallel for
	for(i=0;i<N;i++) {
		if(a[i] > max)
#pragma omp critical
		{
		if(a[i] > max)
			max = a[i];
		}
	}	

	gettimeofday(&t2, NULL);
	t = (t2.tv_sec-t1.tv_sec) + 1e-6*(t2.tv_usec-t1.tv_usec);
	printf("Max = %d, took %f\n", max, t);
}
