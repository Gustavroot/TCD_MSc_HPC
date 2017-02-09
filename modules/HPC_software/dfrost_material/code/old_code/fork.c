#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include <sys/wait.h>


int main() {

	pid_t pid;
	int stat;

	pid = fork();


	switch(pid) {

		case -1:
			printf("Couldn't fork\n");
			exit(0);

		case 0:
			printf("I am the child\n");
			printf("Child: my pid = %d\n", getpid());
			printf("Child: my parent's pid = %d\n", getppid());

			// execl("/bin/bash", "xxxxxxxx", NULL);
			//sleep(5);
			execl("/bin/ls", "ls", "-l", "-a", "/home", NULL);
			
			break;
		default:
			printf("I am the parent. Child pid = %d\n", pid);
			printf("Parent: my pid = %d\n", getpid());
 
			printf("Calling waitpid\n");
			waitpid(pid, &stat, 0);
			printf("Parent returned from wait\n");

			if(WIFEXITED(stat)) {
				printf("Child bailed with status %d\n", WEXITSTATUS(stat));
			}
			// sleep(100);
			break;
	}

	printf("Foooooo\n");
}
