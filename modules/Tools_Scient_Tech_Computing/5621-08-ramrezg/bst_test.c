#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#include "bst.h"

#define N 1000
#define SEED 97


/*
 * Purpose:
 * Display some performance metrics for a binary search tree.
 *
 * The main operation being tested is the insert operation.
 *
 * We compare the time taken to insert values in random order,
 * against the degenerate case (where the values are inserted
 * already in order, and the tree effectively becomes a linked
 * list, and any binary structure is lost).
 */


void usage(char arg0[]) {
	fprintf(stderr, "Usage: %s [-n NUM_TESTS]\n", arg0);
	exit(EXIT_FAILURE);
}


int main(int argc, char *argv[]) {
	/* declare variables */
	bst *my_tree;
	int n = N;
	int i, k;

	/* for getopt */
	int opt;

	/* for gettimeofday */
	struct timeval start, end;
	long elapsed;


	/* seed rng */
	srand48(SEED);


	/* process args */
	while ((opt = getopt(argc, argv, "n:h")) != -1) {
		switch (opt) {
			case 'n':
				n = atoi(optarg);
				break;
			case 'h':
			default: /* '?' */
				usage(argv[0]);
		}
	}


	/**********************************************************************/
	/* PART1 - get timing for some random inserts */
	/**********************************************************************/

	/* start the clock */
	gettimeofday(&start, NULL);

	/* init the tree */
	my_tree = bst_create();

	/* populate the tree with some random ints */
	for (i=0; i<n; i++) {
		k = (int)(n * drand48());

		bst_insert(my_tree, k);
	}

	/* tidy up */
	bst_destroy(my_tree);

	/* stop the clock */
	gettimeofday(&end, NULL);
	elapsed = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) / 1000;	// milliseconds

	/* printf, start... more below */
	printf("%d RANDOM %ld ", n, elapsed);



	/**********************************************************************/
	/* PART2 - get timing for linear inserts */
	/**********************************************************************/

	/* start the clock */
	gettimeofday(&start, NULL);

	/* init the tree */
	my_tree = bst_create();

	/* populate the tree with ints in linear order */
	for (i=0; i<n; i++) {
		bst_insert(my_tree, i);
	}

	/* tidy up */
	bst_destroy(my_tree);

	/* stop the clock */
	gettimeofday(&end, NULL);
	elapsed = ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec)) / 1000;	// milliseconds

	/* printf, continued */
	printf("LINEAR %ld\n", elapsed);



	return(0);
}



/*
 * vim:ts=4:sw=4
 */
