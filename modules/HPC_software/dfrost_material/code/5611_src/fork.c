#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {

	pid_t pid;

	pid = fork();

	if(pid == 0) {
		/* child */
		execl("/bin/ls", "XXXXXX", "-l", "/etc", NULL);
	} else {
		printf("Parent\n");

		sleep(30);
	}

}
