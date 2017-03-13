#include <stdio.h>
#include <getopt.h>
#include <time.h>
#include <stdlib.h>

#include <omp.h>

//#define NUM_THREADS 8


//NOTE: no pivoting is implemented on Gauss
//elimination; testing square matrices only

//Compilation instructions:
//	$ gcc -fopenmp -o gauss-parallel gauss-parallel.c


//EXTRA functions

void print_usage(){
  printf("USAGE: ./gauss -n POSITIVE_INT NR_CORES\n");
}

void print_mat(double* mat, int n, int m){
  int i, j;
  for(i=0; i<n; i++){
    for(j=0; j<m; j++){
      printf("%.5f\t", mat[i*m+j]);
    }
    printf("\n");
  }
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

  printf("From Gauss parallel:\n\n");

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
  double d_t;

  //seeding random numbers generations
  srand(time(NULL));

  //allocating memory for augmented matrix and solution vector
  double* aug_mat = (double*) malloc((N*N+N)*sizeof(double));
  if( aug_mat == NULL ){
    printf("\nERROR: malloc wasn't able to allocate memory for matrix\n\n");
    return 0;
  }
  double* result = (double*) malloc(N*sizeof(double));

  double buff, sum;

  //--------------------------------

  gettimeofday(&begin, NULL);

  //randomly filling aug matrix
  for(i=0; i<N; i++){
    for(j=0; j<N; j++){
      aug_mat[i*(N+1)+j] = (double) rand()/RAND_MAX;
    }
  }

  //randomly filling column of input vector \vec{b}
  j=N;
  for(i=0; i<N; i++){
    aug_mat[i*(N+1)+j] = (double) rand()/RAND_MAX;
  }

  //upper triangular part
  for(j=0; j<=N; j++){
    //parallelizing
    #pragma omp parallel for private(k,buff)
    for(i=0; i<N; i++){
      if(i>j){
        buff=aug_mat[i*(N+1)+j]/aug_mat[j*(N+1)+j];
        for(k=0; k<=N; k++){
          aug_mat[i*(N+1)+k]=aug_mat[i*(N+1)+k]-buff*aug_mat[j*(N+1)+k];
        }
      }
    }
  }

  //solving backwards
  result[N-1] = aug_mat[(N-1)*(N+1)+(N)]/aug_mat[(N-1)*(N+1)+(N-1)];
  for(i=(N-2); i>=0; i--){
    sum=0;
    for(j=i+1; j<N; j++){
      sum = sum+aug_mat[i*(N+1)+j]*result[j];
    }
    result[i]=(aug_mat[i*(N+1)+(N)]-sum)/aug_mat[i*(N+1)+i];
  }

  gettimeofday(&end, NULL);

  //--------------------------------

  d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec -begin.tv_usec)/1000000.0);

  printf("System solved!\n");
  printf("Execution time: %.4f\n", d_t);

  //releasing memory
  free(aug_mat);
  free(result);

  return 0;
}
