#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

//Compilation instructions
//	gcc matmul.c -o matmul

//Execution instructions (N1, N2 and N3 are associated to matrix dimensions)
//	./matmul N1 N2 N3


//Function for matrix multiplication
double mat_mul(double **A, double **B, double **C, int l, int m, int n){
  double diff_time;
  //Definition of variables to store times
  struct timeval begin, end;
  //Start timing here
  gettimeofday(&begin, NULL);
  double buff_summ = 0;
  for(int i=0; i<l; i++){
    for(int j=0; j<n; j++){
      for(int k=0; k<m; k++){
        buff_summ = buff_summ + A[i][k]*B[k][j];
      }
      C[i][j] = buff_summ;
      buff_summ = 0;
    }
  }
  //Stop timing here
  gettimeofday(&end, NULL);
  //Calculating time difference
  diff_time = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);
  return diff_time;
}


//Function to print matrices
void matrix_print(double **M, int n_orig, int m_orig){
  //First, restrict printing a matrix dimension greater than 9
  int n, m;
  if(n_orig>9){n=9;}
  else{n = n_orig;}
  if(m_orig>9){m=9;}
  else{m = m_orig;}
  //Then, printing elements of matrix with double for loop
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      printf("%.3f \t", M[i][j]);
    }
    if(m_orig>9){printf("...\n");}
    else{printf("\n");}
  }
  if(n_orig>9){printf("... \n");}
}


//Function to fill a matrix of nxm with random numbers
void matrix_filling(double **M, int n, int m){
  //First, set the seed for random gen
  srand((unsigned)time(NULL));

  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      //Random numbers between -2 and +2
      M[i][j] = ((double)rand()/(double)RAND_MAX)*4-2;
    }
  }
}


//Function to check if string is int
int check_if_int(char str_buff[]){
  for(int i=0; i<strlen(str_buff); i++){
    if(!isdigit(str_buff[i])){return 0;}
  }
  return 1;
}


int main(int argc, char *argv[]){
  printf("\nProgram to measure code performance in matrix multiplication.\n\n");

  if(argc != 4){
    printf("Incorrect number of parameters.\n");
  }
  else{
    int checker = 0;
    for(int i=1; i<argc; i++){
      if(check_if_int(argv[i]) == 0){checker++;}
    }
    if(checker>0){
      printf("The required params must be integers.\n");
    }
    else{
      //Conversion from strings to int, of sizes for matrices
      int l, m, n;
      l = atoi(argv[1]);
      m = atoi(argv[2]);
      n = atoi(argv[3]);

      //Generating matrices

      double **A;
      //--matrix A
      A = (double **) malloc(l*sizeof(double*));
      for (int i = 0; i < l; i++){
        A[i] = (double *) malloc(m*sizeof(double));
      }
      matrix_filling(A, l, m);
      printf("Matrix A:\n");
      matrix_print(A, l, m);

      double **B;
      //--matrix B
      B = (double **) malloc(m*sizeof(double*));
      for (int i = 0; i < m; i++){
        B[i] = (double *) malloc(n*sizeof(double));
      }
      matrix_filling(B, m, n);
      printf("\nMatrix B:\n");
      matrix_print(B, m, n);

      //Matrices multiplication
      double **C;
      //--matrix C
      C = (double **) malloc(l*sizeof(double*));
      for (int i = 0; i < l; i++){
        C[i] = (double *) malloc(n*sizeof(double));
      }
      long double exec_time = mat_mul(A, B, C, l, m, n);
      printf("\nMatrix C:\n");
      matrix_print(C, l, n);

      printf("\nExecution time: %Le \n", exec_time);

      //Releasing m-allocated memory
      for (int i = 0; i < l; i++){
        free(A[i]);
      }
      free(A);
      for (int i = 0; i < m; i++){
        free(B[i]);
      }
      free(B);
      for (int i = 0; i < l; i++){
        free(C[i]);
      }
      free(C);
    }
  }
  printf("\n");
}
