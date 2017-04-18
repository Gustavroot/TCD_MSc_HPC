#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>

#include <sys/time.h>



//compilation instructions
//	$ gcc -o cyl_rad cyl_rad.c

typedef float VAR_TYPE;


//CUDA functions

extern inline VAR_TYPE cyl_rad_cu(VAR_TYPE*, VAR_TYPE*, VAR_TYPE*, int, int, double*, int, int, int);


//EXTRA functions
void print_usage(){
  printf("./cyl_rad [-n N] [-m M] [-p ITER]\n");
}
void print_matrix(VAR_TYPE* mat, int n, int m){
  int m_buff, n_buff;

  printf("\n");

  int i, j;
  if(n>10){
    n_buff = 10;
    printf("Printing matrix up to size %d rows...\n", n_buff);
  }
  else{n_buff = n;}
  if(m>10){
    m_buff = 6;
    printf("Printing matrix up to size %d cols...\n", m_buff);
  }
  else{m_buff = m;}

  printf("\n");

  for(i=0; i<n_buff; i++){
    for(j=0; j<m_buff; j++){
      printf("%.9f\t\t", mat[i*m + j]);
    }
    printf("\n");
  }
}


//main code
int main(int argc, char** argv){

  //time-measuring variables
  struct timeval begin, end;
  double d_t;
  double *d_t_cuda = &d_t;

  //used by getopt
  int option;

  //general purpose counter
  int i, j, k;

  //pointer to the matrix elems
  VAR_TYPE *M, *M_next;

  int n_rows = 32, m_cols = 32;
  int nr_iterations = 10;

  //extra counter to check params with getopt
  int params_counter = 0;
  int nr_threads_per_block = 1024;

  //flag to average or not temperatures at the end
  char to_average_T = 0;

  //flag to choose if timing
  int to_time = 0;

  //checking input params
  if(argc > 9){
    print_usage();
    return 0;
  }

  //extracting params and flags from args
  while ((option = getopt(argc, argv,"p:n:m:at")) != -1) {
    switch (option) {
      case 't':
        to_time = 1;
        params_counter++;
        break;
      case 'a':
        to_average_T = 1;
        params_counter++;
        break;
      case 'p':
        nr_iterations = atoi(optarg);
        params_counter += 2;
        break;
      case 'n':
        n_rows = atoi(optarg);
        params_counter += 2;
        break;
      case 'm':
        m_cols = atoi(optarg);
        params_counter += 2;
        break;
      default: print_usage();
        printf("ERROR: incorrect input flags.\n");
        return 0;
    }
  }

  //checking number of params
  if( (argc-1) != params_counter ){
    printf("ERROR: wrong flags.\n");
    print_usage();
    return 0;
  }

  if(n_rows%32 != 0 || m_cols%32 != 0){
    printf("\nTry to set n and m divisible by 32.\n\n");
    return 0;
  }

  if(m_cols%nr_threads_per_block != 0){
    printf("\nTry to set m divisible by %d.\n\n", nr_threads_per_block);
    return 0;
  }

  //TODO: check n, m, p values for int-like

  printf("\nSystem specs:\n\n");
  printf("\t** n = %d\n", n_rows);
  printf("\t** m = %d\n", m_cols);
  printf("\t** nr of iterations = %d\n", nr_iterations);

  printf("\n-------------\n\n");

  //initial time measurement
  gettimeofday(&begin, NULL);

  //allocating memory for matrix
  M = (VAR_TYPE*) malloc(n_rows*m_cols*sizeof(VAR_TYPE));
  //and for the aux matrix to calculate the next step
  M_next = (VAR_TYPE*) malloc(n_rows*m_cols*sizeof(VAR_TYPE));

  //end time measurement
  gettimeofday(&end, NULL);

  if(to_time){
    d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
    printf("\nTime taken for allocation in serial: %f\n", d_t);
  }

  if( M == NULL || M_next == NULL){
    printf("\nERROR: malloc wasn't able to allocate memory for matrix\n\n");
    return 0;
  }

  //initial time measurement
  gettimeofday(&begin, NULL);

  //set values for column 0
  j = 0;
  for(i=0; i<n_rows; i++){
    M[i*m_cols + j] = 1.00*(VAR_TYPE)(i+1)/(VAR_TYPE)(n_rows);
    M_next[i*m_cols + j] = 1.00*(VAR_TYPE)(i+1)/(VAR_TYPE)(n_rows);
  }
  //set values for column 1
  j = 1;
  for(i=0; i<n_rows; i++){
    M[i*m_cols + j] = 0.90*(VAR_TYPE)(i+1)/(VAR_TYPE)(n_rows);
    M_next[i*m_cols + j] = 0.90*(VAR_TYPE)(i+1)/(VAR_TYPE)(n_rows);
  }

  //end time measurement
  gettimeofday(&end, NULL);

  if(to_time){
    d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
    printf("\nTime taken for init of matrices: %f\n", d_t);
  }

  //DEBUG print
  print_matrix(M, n_rows, m_cols);

  //initial time measurement
  gettimeofday(&begin, NULL);

  //general loop, where each step represents one
  //time interval over the finite diff method
  int n = n_rows, m = m_cols;
  VAR_TYPE* M_buff;
  printf("\nIteration number: ");
  for(k=0; k<nr_iterations; k++){

    for(i=0; i<n; i++){
      //omit modifications over j = {0, 1}
      for(j=2; j<m; j++){
        //when j equals one in {m-1, m-2}, then apply boundary conditions
        if(j == m-2){
          M_next[i*m + j] = (1/( (VAR_TYPE)(5.0) )) * (1.9*M[i*m + (j-2)] +
                  1.5*M[i*m + (j-1)] + M[i*m + j] + 0.5*M[i*m + (j+1)]
                  + 0.1*M[i*m + 0] );
        }
        else if(j == m-1){
          M_next[i*m + j] = (1/( (VAR_TYPE)(5.0) )) * (1.9*M[i*m + (j-2)] +
                  1.5*M[i*m + (j-1)] + M[i*m + j] + 0.5*M[i*m + 0]
                  + 0.1*M[i*m + 1] );
        }
        else{
          M_next[i*m + j] = (1/( (VAR_TYPE)(5.0) )) * (1.9*M[i*m + (j-2)] +
                  1.5*M[i*m + (j-1)] + M[i*m + j] + 0.5*M[i*m + (j+1)]
                  + 0.1*M[i*m + (j+2)] );
        }
      }
    }

    //swap of matrices
    M_buff = M;
    M = M_next;
    M_next = M_buff;

    //M = one_step_fin_diff(M, M_next, n_rows, m_cols);

    //printf("serial M[2] = %f\n", M[2]);
    if(k%5 == 0){
      printf("%d... ", k+1);
    }
  }
  printf("\n\n");

  //end time measurement
  gettimeofday(&end, NULL);

  if(to_time){
    d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
    printf("\nEXEC TIME: %f\n", d_t);
  }

  //DEBUG print
  printf("\nAfter evolution of the system:\n\n");
  print_matrix(M, n_rows, m_cols);

  //if average T flag was passed
  if(to_average_T){
    VAR_TYPE T;
    printf("\nAverage of temperatures:\n\n");
    for(i = 0; i<n_rows; i++){
      T = 0.0;
      for(j = 0; j<m_cols; j++){
        T += M[i*m_cols + j];
      }
      printf("row %4d: %.10f\n", i+1, T/ ( (VAR_TYPE)(m_cols) ));
      if(i>11){
        printf("...\n\n");
        break;
      }
    }
  }

  printf("\n-----------------------\n\n");

  //parallel implementation
  printf("Adding parallel computation:\n\n");

  //re-set the values of the matrices
  //set values for column 0
  j = 0;
  for(i=0; i<n_rows; i++){
    M[i*m_cols + j] = 1.00*(VAR_TYPE)(i+1)/(VAR_TYPE)(n_rows);
    M_next[i*m_cols + j] = 1.00*(VAR_TYPE)(i+1)/(VAR_TYPE)(n_rows);
  }
  //set values for column 1
  j = 1;
  for(i=0; i<n_rows; i++){
    M[i*m_cols + j] = 0.90*(VAR_TYPE)(i+1)/(VAR_TYPE)(n_rows);
    M_next[i*m_cols + j] = 0.90*(VAR_TYPE)(i+1)/(VAR_TYPE)(n_rows);
  }
  //and for the other columns
  for(i=0; i<n_rows; i++){
    for(j=2; j<m_cols; j++){
      M[i*m_cols + j] = 0.0;
      M_next[i*m_cols + j] = 0.0;
    }
  }

  //allocating memory for thermostat
  VAR_TYPE* thermostat = (VAR_TYPE*) malloc(n_rows*sizeof(VAR_TYPE));

  //initial time measurement
  gettimeofday(&begin, NULL);

  cyl_rad_cu(M, M_next, thermostat, n_rows, m_cols, d_t_cuda, to_time, nr_iterations, nr_threads_per_block);

  //end time measurement
  gettimeofday(&end, NULL);

  if(to_time){
    d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
    printf("\nEXEC TIME, parallel part (total): %f\n\n", d_t);
  }

  //DEBUG print
  printf("After parallel computations:\n\n");
  print_matrix(M, n_rows, m_cols);

  //if average T flag was passed
  if(to_average_T){
    printf("\nAverage of temperatures:\n\n");
    for(i = 0; i<n_rows; i++){
      printf("row %4d: %.10f\n", i+1, thermostat[i] / ( (VAR_TYPE)(m_cols) ));
      if(i>11){
        printf("...\n\n");
        break;
      }
    }
  }

  free(M);

  printf("\n");

  return 0;
}