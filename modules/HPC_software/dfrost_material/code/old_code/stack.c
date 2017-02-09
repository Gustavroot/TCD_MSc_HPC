#include <stdio.h>

void f();

int main() {
	// f();
	//
	int *x;

	x = malloc(10000* sizeof(int));

	for(i=0 ...)
		x[i] = random();
}

void f() {

	int x[10000];
	static int i;
	i++;
	printf("%d\n", i);
	f();
	return;
}
