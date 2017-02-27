#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>


//compilation instructions
//	$ gcc mat_norms_serial.c -lm


//CORE functions

//different ways of calculating norms

float max_norm(float*, int, int);
float frobenius_norm();
float one_norm(float*, int, int);
float infinite_norm(float*, int, int);

void print_matrix(float*, int, int);


//EXTRA functions
void print_usage(){
  printf("./mat_norms_serial [-n N] [-m M] [-s] [-t]\n");
}



int main(int argc, char** argv){

  //time-measuring variables
  struct timeval begin, end;
  double d_t;

  //used by getopt
  int option;
  
  //general purpose counter
  int i;
  
  //pointer to the matrix elems
  float* M;
  
  int n_rows = 10, m_cols = 10;
  time_t seed = 123456;
  int to_time = 0;
  
  //buffer to store the different norms
  float norm_buff;
  
  int params_counter = 0;

  //checking input params
  if(argc > 7){
    printf("ERROR: number of input params not allowed.\n");
    return 0;
  }

  //extracting params and flags from args
  while ((option = getopt(argc, argv,"stn:m:")) != -1) {
    switch (option) {
      case 's':
        //setting seed to number of milliseconds
        seed = time(NULL)*1000;
        params_counter++;
        break;
      case 't':
        to_time = 1;
        params_counter++;
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
    printf("ERROR: wrong input params.\n");
    print_usage();
    return 0;
  }
  
  //TODO: check here if n_rows, n_cols and seed have the
  //appropriate values
  
  //setting seed
  srand48(seed);
  
  //allocating memory for matrix
  M = (float*) malloc(n_rows*m_cols*sizeof(float));
  
  if( M == NULL ){
    printf("\nERROR: malloc wasn't able to allocate memory for matrix\n\n");
    return 0;
  }
  
  //initializing matrix
  for(i=0; i<n_rows*m_cols; i++){
    M[i] = (float)(drand48());
  }
  
  //DEBUG print
  //print_matrix(M, n_rows, m_cols);
  
  printf("\n");
  
  //matrix norm
  //initial time measurement
  gettimeofday(&begin, NULL);
  norm_buff = max_norm(M, n_rows, m_cols);
  //end time measurement
  gettimeofday(&end, NULL);
  printf("max norm: %f\n", norm_buff);
  if(to_time){
    d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
    printf("*** execution time for max norm: %f\n", d_t);
  }

  printf("\n");
  
  //frobenius norm
  //initial time measurement
  gettimeofday(&begin, NULL);
  norm_buff = frobenius_norm(M, n_rows, m_cols);
  //end time measurement
  gettimeofday(&end, NULL);
  printf("Frobenius norm: %f\n", norm_buff);
  if(to_time){
    d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
    printf("*** execution time for Frobenius norm: %f\n", d_t);
  }

  printf("\n");

  //one norm
  //initial time measurement
  gettimeofday(&begin, NULL);
  norm_buff = one_norm(M, n_rows, m_cols);
  //end time measurement
  gettimeofday(&end, NULL);
  printf("one norm: %f\n", norm_buff);
  if(to_time){
    d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
    printf("*** execution time for one norm: %f\n", d_t);
  }

  printf("\n");

  //infinite norm
  //initial time measurement
  gettimeofday(&begin, NULL);
  norm_buff = infinite_norm(M, n_rows, m_cols);
  //end time measurement
  gettimeofday(&end, NULL);
  printf("infinite norm: %f\n", norm_buff);
  if(to_time){
    d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
    printf("*** execution time for infinite norm: %f\n", d_t);
  }

  return 0;
}




float max_norm(float* M, int n, int m){
  float abs_val = 0;
  int i, j;
  //looping over the integral to get the max abs value
  for(i=0; i<n; i++){
    for(j=0; j<m; j++){
      if( fabs(M[i*m + j]) > abs_val ){
        abs_val = fabs(M[i*m + j]);
      }
    }
  }
  return abs_val;
}

float frobenius_norm(float* M, int n, int m){
  float norm = 0;
  int i, j;
  
  for(i=0; i<n; i++){
    for(j=0; j<m; j++){
      norm += pow(M[i*m + j], 2);
    }
  }
  
  norm = sqrt(norm);

  return norm;
}

float one_norm(float* M, int n, int m){
  int i, j;
  
  float norm = 0, norm_buff = 0;
  
  for(j=0; j<m; j++){
    for(i=0; i<n; i++){
      norm_buff += fabs(M[i*m + j]);
    }
    if(norm_buff > norm){
      norm = norm_buff;
    }
    norm_buff = 0;
  }

  return norm;
}

float infinite_norm(float* M, int n, int m){
  int i, j;
  
  float norm = 0, norm_buff = 0;
  
  for(i=0; i<m; i++){
    for(j=0; j<n; j++){
      norm_buff += fabs(M[i*m + j]);
    }
    if(norm_buff > norm){
      norm = norm_buff;
    }
    norm_buff = 0;
  }

  return norm;
}



void print_matrix(float* M, int n, int m){
  int i, j;
  for(i=0; i<n; i++){
    for(j=0; j<m; j++){
      printf("%.2f\t", M[i*n + j]);
    }
    printf("\n");
  }
}
