#include <stdio.h>
#include <pthread.h>

pthread_mutex_t lock;
int sum=0;
int lsum[2];

void * foo(void *p) {

	int i;

	for(i=0;i<5;i++) {
		pthread_mutex_lock(&lock);
		printf("FOO got lock\n");
		// sleep(1);
		pthread_mutex_unlock(&lock);
		// sleep(1);
/*
		pthread_mutex_lock(&lock);
		sum += random()%100;
		pthread_mutex_unlock(&lock);
*/
		lsum[0] += random()%100;
	}
	pthread_mutex_lock(&lock);
	sum += lsum[0];
	pthread_mutex_unlock(&lock);

	return;
}

void * bar(void *p) {

	int i;

	for(i=0;i<5;i++) {
		pthread_mutex_lock(&lock);
		printf("BAR got lock\n");
		// sleep(1);
		pthread_mutex_unlock(&lock);
		// sleep(1);
		// pthread_mutex_lock(&lock);
		// sum += random()%100;
		// pthread_mutex_unlock(&lock);
		lsum[1] += random()%100;
	}
	pthread_mutex_lock(&lock);
	sum += lsum[1];
	pthread_mutex_unlock(&lock);

	return;
}

int main() {
	pthread_t handle[2];
	int x[2];
	x[0] = 0;
	x[1] = 1;

	pthread_mutex_init(&lock, NULL);
	printf("Sum = %d\n", sum);

	pthread_create(&handle[0], NULL, foo, NULL);
	pthread_create(&handle[1], NULL, bar, NULL);

	pthread_join(handle[0], NULL);	
	pthread_join(handle[1], NULL);	

	printf("Sum = %d\n", sum);
	return 0;
}
