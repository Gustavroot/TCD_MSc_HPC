#include <stdio.h>
#include <omp.h>

int main() {

#pragma omp parallel
{
	int id;
	id = omp_get_thread_num();
	printf("%d: Hello world\n", id);

	if(id == 0) {
		/* Master */
	} else {
		/* Slave */
	}
}

	omp_set_num_threads(3);

#pragma omp parallel num_threads(6)
{
	int id;
	id = omp_get_thread_num();
	printf("%d: Hello world again\n", id);

}

#pragma omp parallel
{
	int id;
	id = omp_get_thread_num();
	printf("%d: Hello world yet again\n", id);
}

return 0;
}
