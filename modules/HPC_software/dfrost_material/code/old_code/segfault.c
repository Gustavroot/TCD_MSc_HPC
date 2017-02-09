#include <stdlib.h>
#include <stdio.h>

#define N 100

int f() {

	static int i=0;
	double x[1000];

	printf("Hello from fucntion f() %d\n", i++);

	f();
}

int main() {

	void *p;
	int i;
	int NN;

	for(i=0;i<N;i++) {
		printf("N Hello\n");
	}

	printf("Hello from main\n");
	f();
	p = malloc(20);
	printf("Goodbye from main\n");
}
