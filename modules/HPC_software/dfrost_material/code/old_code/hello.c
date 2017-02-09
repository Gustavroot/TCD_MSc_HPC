#include <stdio.h>
#include <stdlib.h>

#define N 1000

int f(double X[N]) {

	return 0;

}

int g(double *p) {

	int i;
	for(i=0;i<N;i++) {
		// p[i] = i;
	}
	return 0;
}

int main() {

	double *x;

	x = (double *)malloc(N*sizeof(double));
	if(x == NULL) {
		printf("Couldn't malloc x\n");
		exit(1);
	}


	printf("x[2] = %f\n", x[2]);
	g(x);
	printf("x[2] = %f\n", x[2]);


	free(x);
	return 2;

}
