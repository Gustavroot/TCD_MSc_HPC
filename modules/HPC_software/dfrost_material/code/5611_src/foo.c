#include <stdio.h>
#include <stdlib.h>

void foo() {

	// int x[1000];
	
	int *x;
	x = malloc(1000*sizeof(int));

	x[999] = random();

	printf("%d\n", x[999]);
	foo();
}

int main() {

	int x, y, z;


	printf("%x %x %x\n", &x, &y, &z);

	foo();

}
