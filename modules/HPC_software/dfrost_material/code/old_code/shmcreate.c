#include<stdio.h>
#include<unistd.h>
#include<sys/ipc.h>
#include<sys/shm.h>

#define SHM_KEY 99

int main() {

	int shmid;
	int *addr, *addr2;

	/* Create a 64k region */
	shmid = shmget(SHM_KEY, 64*1024, 0777|IPC_CREAT);

	/* Attach to the region twice */
	addr = shmat(shmid, 0, 0);
	addr2 = shmat(shmid, 0, 0);

	printf("Region attached at %x and %x\n", addr, addr2);

	/* Update the region with one pointer and read using the other */
	addr[0] = 77;
	printf("addr2[0] = %d\n", addr2[0]);

	sleep(10000);

	/* Remove the region */
	shmctl(shmid, IPC_RMID, 0);

	return 0;
}
