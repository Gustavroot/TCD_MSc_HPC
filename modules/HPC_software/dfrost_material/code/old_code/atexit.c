#include <stdio.h>
#include <stdlib.h>


void foo() {
	printf("Running foo\n");
	exit(2);
	printf("Ignored exit in foo\n");
}

void bar() {
	printf("Running bar\n");
}

int main() {

	atexit(bar);
	atexit(foo);

	exit(2);
}
