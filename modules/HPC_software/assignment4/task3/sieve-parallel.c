#include <stdio.h>
#include <getopt.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h>


#define PRINT_LIMIT 80
//#define NUM_THREADS 8


//NOTE: no pivoting is implemented on Gauss
//elimination; testing square matrices only

//Compilation instructions:
//	$ gcc -fopenmp -lm -o gauss gauss.c


//EXTRA functions

void print_usage(){
  printf("USAGE: ./gauss -n POSITIVE_INT NR_CORES\n");
}



//main code
int main(int argc, char** argv){

  //general-purpose counters
  int i, j, k;

  int option, N = -1;

  if(argc != 4){
    print_usage();
    return 0;
  }

  int NUM_THREADS = atoi(argv[3]);

  omp_set_num_threads(NUM_THREADS);

  //getting value of N with getopt
  while ((option = getopt(argc, argv,"n:")) != -1) {
    switch (option) {
      case 'n' :
        N = atoi(optarg);
        break;
      default:
        print_usage();
        return 0;
    }
  }

  //check that N val is a positive integer
  if(N <= 0){
    printf("ERROR: N must be > 0.\n");
    return 0;
  }

  //time-measuring variables
  struct timeval begin, end;
  double d_t, d_t_cu;

  //allocating array needed for algorithm
  int* numbers = (int*) malloc( N*sizeof(int) );
  if( numbers == NULL ){
    printf("\nERROR: malloc wasn't able to allocate memory for matrix\n\n");
    return 0;
  }

  #pragma omp parallel for
  //initially, all considered primes
  for(i=0; i<N; i++){
    numbers[i] = 1;
  }

  //--------------------------

  gettimeofday(&begin, NULL);

  // find all non-primes
  #pragma omp parallel for schedule(dynamic) private(j)
  for(i=2; i <= (int)sqrt((double)N); i++){
    if (numbers[i]){
      for(j = i*i; j <= N; j += i){
        numbers[j] = 0;
      }
    }
  }

  gettimeofday(&end, NULL);

  //--------------------------------

  d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -begin.tv_usec)/1000000.0);

  printf("Execution time: %.4f\n\n", d_t);

  //printing results
  k=0;
  for(i=2; i<N; i++){
    if(numbers[i] == 1){
      printf("%d\t", i);
      k++;
      if(k%8 == 0){
        printf("\n");
      }
    }
    if( k > PRINT_LIMIT ){
      printf("\n\nand many more...");
      break;
    }
  }

  printf("\n");

  //releasing memory
  free(numbers);

  return 0;
}
