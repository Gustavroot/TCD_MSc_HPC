#include <sys/types.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


void handler(int x) {

	printf("This is the handler with signal %d\n", x);
}

int main() {


	pid_t pid;
	int i;

	pid = fork();


	if(pid) {
		/* Parent */
		sleep(1);
		for(i=1;i<100;i++) {
			if(kill(pid, i) != 0) {
				printf("Kill failed\n");
				exit(0);
			}
			sleep(1);
		}

	} else { 
		/* Child */
		for(i=1;i<30;i++)
			signal(i, handler);

		while(1) {
		}
	}


}
