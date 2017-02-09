#include <stdio.h>
#include <signal.h>

void handler(int x) {
	printf("Recv'd signal %d\n", x);
	return;
}

int main() {

	int i;

	for(i=1;i<30;i++) {
		signal(i, handler);
	}	


	while(1) {
	}
}
