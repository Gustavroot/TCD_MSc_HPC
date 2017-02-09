/*
 * A threaded program that has three ways of calculating PI
 * 
 * gcc pthread.c -lgslcblas -lgsl -lpthread
 *    This version lets main sum up the values
 *
 * gcc -DMUTEX pthread.c -lgslcblas -lgsl -lpthread
 *    This version uses a mutex and lets each thread contrib to total
 *
 * gcc -DBADMUTEX pthread.c -lgslcblas -lgsl -lpthread
 *    This version uses a mutex and lets each thread contrib to total
 *    but at the wrong location
 *
 * Run 'time ./a.out' for various numbers of threads and problem sizes
 * and compare real, user and sys times for each code version
 * 
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>

#include <sys/time.h>
#include <time.h>


pthread_mutex_t mylock;
pthread_mutex_t mylock2;
pthread_mutex_t mylock3;
pthread_mutex_t mylock4;
pthread_mutex_t mylock5;
long total_hits;

int num_iter;
int *hits;

void * foo(void *arg) {

	struct timeval now;
	int *p_num;
	int i, hit_count;
	const gsl_rng_type * T; gsl_rng * r;
	double x, y, dist;

	p_num = (int *)arg;
	gettimeofday(&now, NULL);

	gsl_rng_env_setup();
	T = gsl_rng_default;
	r = gsl_rng_alloc(T);
	gsl_rng_set(r, *p_num+now.tv_sec+now.tv_usec);

	/* Calculate PI!! */
	hit_count = 0;
	for(i=0;i<num_iter;i++) {
		x = 2*(gsl_rng_uniform(r)-0.5);
		y = 2*(gsl_rng_uniform(r)-0.5);
		dist = x*x + y*y;
		if(dist < 1) {
			hit_count++;
#ifdef BADMUTEX
			pthread_mutex_lock(&mylock);
			// total_hits += hit_count;
			total_hits++;
			pthread_mutex_unlock(&mylock);
#endif
	
		}
	}

#ifdef MUTEX
	pthread_mutex_lock(&mylock);
	total_hits += hit_count;
	pthread_mutex_unlock(&mylock);
#else
	hits[*p_num] = hit_count;	
#endif

	return NULL;
}

int main() {

	pthread_t *handles;
	int *thread_args;
	int i, num_threads;

	pthread_mutex_init(&mylock, NULL);
	total_hits = 0;
	num_threads = 40;
	num_iter = 1000000;

	handles = (pthread_t *)malloc(num_threads*sizeof(pthread_t));
	thread_args = (int *)malloc(num_threads*sizeof(int));
	hits = (int *)malloc(num_threads*sizeof(int));

	for(i=0;i<num_threads;i++) {
		thread_args[i] = i;
		pthread_create(&handles[i], NULL, foo, (void *)&thread_args[i]);
// WRONG!!		pthread_create(&handles[i], NULL, foo, (void *)&i);
	}


	for(i=0;i<num_threads;i++) {
		pthread_join(handles[i], NULL);
	}

#ifndef MUTEX
	total_hits = 0;
	for(i=0;i<num_threads;i++)
		total_hits += hits[i];
#endif

	printf("PI = %2.10f\n", (4.0 * total_hits) / (num_threads * num_iter));
	printf("(real) PI = %2.10f\n", M_PI);


	return 0;
}
