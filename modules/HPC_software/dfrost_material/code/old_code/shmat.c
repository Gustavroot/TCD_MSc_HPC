
#include<stdio.h>
#include<stdlib.h>
#include<sys/ipc.h>
#include<sys/shm.h>

#define SHM_KEY 99

int main() {

	int shmid;
	int *addr;

	/* find the id for the region */
	shmid = shmget(SHM_KEY, 0, 0);
	if(shmid == -1) {
		printf("No region to attach to\n");
		exit(1);
	}

	/* Attach to the region */
	addr = shmat(shmid, 0, 0);

	/* Read from the region */
	printf("addr[0] = %d\n", addr[0]);

	return 0;
}
