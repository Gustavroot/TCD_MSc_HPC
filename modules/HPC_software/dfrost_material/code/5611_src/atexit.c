#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void foo() {
	printf("this is foo\n");
	exit(33);
}

void bar() {
	printf("this is bar\n");
	exit(11);
}


int main() {

	atexit(bar);
	atexit(foo);
	atexit(bar);
	atexit(bar);
	atexit(bar);
	atexit(bar);


	atexit(bar);

	foo();
	exit(88);
}
