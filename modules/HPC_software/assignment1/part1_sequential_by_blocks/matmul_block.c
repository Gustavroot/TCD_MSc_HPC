#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <time.h>

//Compilation instructions
//	gcc matmul.c -o matmul

//Execution instructions (N1, N2 and N3 are associated to matrix dimensions)
//..and q, s, r are the nr of blocks in which the dims of matrices are to be divided
//	./matmul N1 N2 N3 q s r


int check_if_int(char str_buff[]);

//most important functions:
void mat_mul(double *A, double *B, double *C, int l, int m, int n, int q, int s, int r);
void mat_mul_blocks(double *A, double *B, double *C, int l,
			int m, int n, int q, int s, int r, int pos_row_C, int pos_col_C);
void matrix_print(double *M, int n_orig, int m_orig);
void matrix_filling(double *M, int n, int m);


int main(int argc, char *argv[]){

  //Giving seed for posterior use of random numbers
  srand(time(NULL));

  printf("\nProgram for matrix multiplication using blocks.\n\n");

  if(argc != 7){
    printf("Incorrect number of parameters.\n");
  }
  else{
    int checker = 0;
    int i, j;
    for(i=1; i<argc; i++){
      if(check_if_int(argv[i]) == 0){checker++;}
    }
    if(checker>0){
      printf("The required params must be integers.\n");
    }
    else{
      //Conversion from strings to int, of sizes for matrices
      int l, m, n, q, s, r;
      l = atoi(argv[1]);
      m = atoi(argv[2]);
      n = atoi(argv[3]);
      q = atoi(argv[4]);
      s = atoi(argv[5]);
      r = atoi(argv[6]);

      //Require here that l/q, m/s, n/r are all ints
      if( l%q!=0 || m%s!=0 || n%r!=0 ){
        printf("The required params for subdivision must divide the matrices dimensions\n");
        return 0;
      }

      //Generating matrices
      double *A;
      A = (double *) malloc( l*m * sizeof(double));
      matrix_filling(A, l, m);
      printf("Matrix A:\n");
      matrix_print(A, l, m);

      double *B;
      B = (double *) malloc( m*n * sizeof(double));
      matrix_filling(B, m, n);
      printf("\nMatrix B:\n");
      matrix_print(B, m, n);

      double *C;
      C = (double *) malloc( l*n * sizeof(double));

      //Before the multiplication step, matrix B is converted into
      //  matrix B_prime, which is easily used for the summation
      //  part of the blocks
      //..the data in B_prime is vertical, and not horizontal like B
      double *B_prime;
      B_prime = (double *) malloc( m*n * sizeof(double));

      int buff_counter = 0;
      for(j=0; j<n; j++){
        for(i=0; i<m; i++){
          B_prime[i+j*m] = B[i*n+j];
        }
      }

      printf("\nRotated matrix (B_prime):\n");
      matrix_print(B_prime, m, n);

      //The multiplication step:
      printf("\n--------------\n");
      mat_mul(A, B_prime, C, l, m, n, q, s, r);
      printf("\nResulting matrix C:\n");
      matrix_print(C, l, n);

      //Releasing m-allocated memory
      free(A);
      free(B);
      free(B_prime);
      free(C);
    }
  }
  printf("\n");

  return 0;
}


//IMPLEMENTATION OF FUNCTIONS:

//Function to check if string is int
int check_if_int(char str_buff[]){
  for(int i=0; i<strlen(str_buff); i++){
    if(!isdigit(str_buff[i])){return 0;}
  }
  return 1;
}

//Function to fill a matrix of nxm with random numbers
void matrix_filling(double *M, int n, int m){
  for(int i=0; i<n*m; i++){
    //Random numbers between -2 and +2
    M[i] = ((double)rand()/(double)RAND_MAX)*4-2;
  }
}

//Function to print matrices
void matrix_print(double *M, int n_orig, int m_orig){
  //First, restrict printing a matrix dimension greater than 9
  int max_size = 15;
  int n, m;
  if(n_orig>max_size){n=max_size;}
  else{n = n_orig;}
  if(m_orig>max_size){m=max_size;}
  else{m = m_orig;}
  //Then, printing elements of matrix with double for loop
  for(int i=0; i<n; i++){
    for(int j=0; j<m; j++){
      printf("%.2f \t", M[i*m+j]);
    }
    if(m_orig>max_size){printf("...\n");}
    else{printf("\n");}
  }
  if(n_orig>max_size){printf("... \n");}
}

//Function for matrix multiplication
//..this function is in charge of multiplying and summing at
//  the same time, hence obtaining one block matrix of C
void mat_mul(double *A, double *B, double *C, int l, int m, int n, int q, int s, int r){
  double buff_summ = 0;
  int i, j;

  for(i=0; i<l; i += q){
    for(j=0; j<n; j += r){
      //each one of the processes/multiplications for each sub-block in C
      mat_mul_blocks(A, B, C, l, m, n, q, s, r, i, j);
    }
  }
}

//One to one correspondence between the following function and
//each block sub-matrix in C
void mat_mul_blocks(double *A, double *B, double *C, int l, int m, int n, int q, int s, int r, int pos_row_C, int pos_col_C){
  int i, j, k;
  double buff_sum;

  //the first 2 fors are for iterating over the elements of Cxy
  for(i=0; i<q; i++){
    for(j=0; j<r; j++){
      //the third for is for the inner product
      buff_sum = 0;
      for(k=0; k<m; k++){
        buff_sum += A[pos_row_C*m + i*m + k]*B[pos_col_C*m + j*m + k];
      }
      //and after the inner product, the Cxy element is assigned
      C[(pos_row_C*n + pos_col_C) + (i*n + j)] = buff_sum;
      //printf("\n C element: (%d, %d), and real pos: %d\n", pos_row_C+i, pos_col_C+j, (pos_row_C*n + pos_col_C) + (i*n + j));
    }
  }
}
