#include <stdio.h>

double sin(double x) {
	
	pid_t pid;

	pid = getpid();

	kill(pid, SIGKILL);

	printf("This is evil sin\n");
	return drand48();

}

int foo() {

	printf("this is my foo function\n");
	return 0;

}
