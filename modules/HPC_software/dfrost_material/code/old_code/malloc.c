#include <stdio.h>
#include <stdlib.h>

int main() {

	// Declare my variables
	double *p, *q, dp;
	int i, n;
	
	n = 258;

	/* allocate my vectors */

	/* 
 	 * This is a really long comment
 	 * 	
 	 * 	sdlfkjsd af hsdafouihs adpfuio hsadfui 
 	 * 	sadfois hadfoisahd fpusadh f
 	 *
 	 * 	sadoufh spadiuf hpsadiu fh
 	 *
 	 *
 	 */

	p = (double *)malloc(n*sizeof(double));
	q = (double *)malloc(n*sizeof(double));
	
	if(p == NULL || q == NULL) {
		printf("Malloc failed\n");
		exit(1);
	}

	for(i=0;i<=10000*n;i++) {
		p[i] = drand48();
		q[i] = drand48();
	}

	dp = 0;
	for(i=0;i<=5*n;i++) {
		dp += p[i]*q[i];
	}

	printf("DP = %f\n", dp);
// 	free(p);
	//free(q);
	return(0);
}















