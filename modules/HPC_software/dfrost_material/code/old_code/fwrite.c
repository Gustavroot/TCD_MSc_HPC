#include <stdio.h>
#include <stdlib.h>

int main() {

	int A[100];
	int B[1000];
	FILE *fp, *fp2;
	int i, j, k, offset, val;

	for(i=0;i<100;i++)
		A[i] = i;

	fp = fopen("intfile", "w");
	fp2 = fopen("intfile-ascii.txt", "w");

	fwrite(A, sizeof(int), 100, fp);
	for(i=0;i<100;i++) {
		fprintf(fp2, "%d\n", A[i]);
	}

	exit(0);

	for(i=0;i<4;i++) {
		for(j=0;j<25;j++) {
			for(k=0;k<10;k++) {
				offset = 40*j + 10*i + k;
				val = 1000*j + 100*k + i;
				B[offset] = val;
			}
		}
	}

	fwrite(B, sizeof(int), 1000, fp);
	
	fclose(fp);

}
