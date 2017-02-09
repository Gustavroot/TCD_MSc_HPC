#include <stdio.h>
#include <stdlib.h>

#define N 10000000

void f3() {
	// int b[N];
	int *b;
	b = malloc(N*sizeof(int));
	printf("Never going to get here\n");
	free(b);
}

void f2() {
	int b[1000];
	printf("hello from f2()\n");
	fflush(stdout);
	f3();
}
	

int main() {

	int a[1000];
	int i;
	printf("Hello from main\n");
	fflush(stdout);


	for(i=0;i<100;i++) {
		a[i] = i;
	}
	f2();
}
