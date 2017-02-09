#include <stdlib.h>

int main() {

	int i, j;
	double *p;

	for(i=1;i<1000;i++) {
		p = malloc(i*1024*sizeof(double));
		for(j=0;j<i*1024;j++) {
			p[j] = 2;
		}
	}

}
