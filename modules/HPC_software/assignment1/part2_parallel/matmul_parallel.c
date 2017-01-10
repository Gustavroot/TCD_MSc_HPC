#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

//Compilation instructios:
//	mpicc matmul_parallel.c -o matmul_parallel -lm (!mpicc to re-compile)


//Execution instructions: (N is the number of processes)
//	mpirun -np N matmul_parallel l m n q s r
//example:
//	mpirun -np 6 matmul_parallel 10 14 12


//IMPORTANT: running in Ubuntu 16.04. To activate
//THE LAM/MPI runtime environment, run "lamboot", and
//for installing the module:
//	sudo apt install lam4-dev

int check_if_int(char str_buff[]);
int best_divisor_q(int num_procs, int l, int n);

//most important functions:
void mat_mul(double *A, double *B, double *C, int l, int m, int n, int q, int s, int r);
void mat_mul_blocks(double *A, double *B, double *C, int l,
			int m, int n, int q, int s, int r, int pos_row_C, int pos_col_C);
void matrix_print(double *M, int n_orig, int m_orig);
void matrix_filling(double *M, int n, int m);


//double portion_B[100000];

//Main code:
void main(int argc, char **argv){

  int m;
  m = atoi(argv[2]);

  int my_id, root_process, num_procs, proc_id;
  int buff_recv_elems;

  MPI_Status status;

  //Root process
  root_process = 0;

  MPI_Init(&argc, &argv);

  /* find out MY process ID, and how many processes were started. */

  MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  if(my_id == root_process){
    printf("\nParallel implementation of matrix multiplication.\n\n");

    int start_index, i, j, buff_num_elems;

    //extra params, for the creation and segmentation of matrices:
    int l, n, q, s, r;
    l = atoi(argv[1]);
    n = atoi(argv[3]);
    //the subdivision dimensions are obtained from matrices dimensions
    //and from the number of processes:
    q = best_divisor_q(num_procs, l, n);
    if(num_procs%2 != 0){printf("The dimensions and np are not appropriate.\n");}
    else if(q == -1){
      printf("The dimensions and np are not appropriate.\n");
    }
    //In case 'almost all good' with dimensions, send data to other processes
    else{

      //Display of specifications of the grid
      printf("Size of the 2D grid of processes: %dx%d\n", q, num_procs/q);

      //setting the correct values of the dividers (q, r)
      r = num_procs/q;
      q = l/q; //because q is not a divisor, but a number of rows
      r = n/r;

      //Require here that l/q, n/r are both ints
      if( l%q!=0 || n%r!=0 ){
        printf("The required params for subdivision must divide the matrices dimensions\n");
      }
      else{

        //Creation of all the data to be used
        //..generating matrices:
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

        //Before sending the data, matrix B is converted into
        //  matrix B_prime, which is easily used for the summation
        //  part in each process
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
        printf("\n\n\n");

        int proc_counter;
        proc_counter = 1;
        //distribute portions of matrices A and B to the slave processes
        for(i=0; i<l; i+=q){
          for(j=0; j<n; j += r){
            if(i==0 && j==0){continue;}

            //send matrix A
            buff_num_elems = q*m;
            start_index = i*m;
            MPI_Send( &buff_num_elems, 1, MPI_INT, proc_counter, 0, MPI_COMM_WORLD);
            MPI_Send( A+start_index, buff_num_elems, MPI_DOUBLE, proc_counter, 0, MPI_COMM_WORLD);

            //send matrix B
            buff_num_elems = r*m;
            start_index = j*m;
            MPI_Send( &buff_num_elems, 1, MPI_INT, proc_counter, 0, MPI_COMM_WORLD);//fix
            MPI_Send( B_prime+start_index, buff_num_elems, MPI_DOUBLE, proc_counter, 0, MPI_COMM_WORLD);

            proc_counter++;
            if(proc_counter%num_procs == 0){break;}
          }
        }

        //the root process also has memory and work assigned
        double buff_sum;
        int k;
        //the first 2 fors are for iterating over the elements of Cxy
        for(i=0; i<q; i++){
          for(j=0; j<r; j++){
            //the third for is for the inner product
            buff_sum = 0;
            for(k=0; k<m; k++){
              buff_sum += A[i*m + k]*B_prime[j*m + k];
            }
            //and after the inner product, the Cxy element is assigned
            C[i*n + j] = buff_sum;
          }
        }

        //buffer array to collect data from slave processes
        double *array2 = (double *)malloc(((l*n)/(num_procs))*sizeof(double));
        //collecting data from slave processes, and stored in array2
        int a, b;
        proc_id = 0;
        for(a=0; a<l; a += q){
          for(b=0; b<n; b += r){
            if(a==0 && b==0){continue;}
            proc_id++;
            if(proc_id == num_procs){break;}
            //IMPORTANT: in the following line, it's 'array2' and not '&array2'
            MPI_Recv( array2, (l*n)/(num_procs), MPI_DOUBLE, proc_id, 1, MPI_COMM_WORLD, &status);
            //the first 2 fors are for iterating over the elements of Cxy
            for(i=0; i<q; i++){
              for(j=0; j<r; j++){
                //and after the inner product, the Cxy element is assigned
                C[(a*n + b) + (i*n + j)] = array2[i*r + j];
              }
            }
          }
        }

        //printing the resulting matrix C
        printf("\nMatrix C:\n");
        matrix_print(C, l, n);

        free(A);
        free(B);
        free(B_prime);
        free(C);

        printf("\n\n");
      }
    }
  }
  else{

    int size_rows, size_cols;
    //int m = atoi(argv[2]);

    //receiving A
    MPI_Recv( &buff_recv_elems, 1, MPI_INT, root_process, 0, MPI_COMM_WORLD, &status);
    size_rows = buff_recv_elems/m;
    double *portion_A = (double *)malloc(buff_recv_elems*sizeof(double));
    MPI_Recv( portion_A, buff_recv_elems, MPI_DOUBLE, root_process, 0, MPI_COMM_WORLD, &status);

    //receiving B
    MPI_Recv( &buff_recv_elems, 1 , MPI_INT, root_process, 0, MPI_COMM_WORLD, &status);
    size_cols = buff_recv_elems/m;
    double *portion_B = (double *)malloc(buff_recv_elems*sizeof(double));
    MPI_Recv( portion_B, buff_recv_elems, MPI_DOUBLE, root_process, 0, MPI_COMM_WORLD, &status);  

    //output array:
    double *array3 = (double *)malloc((size_rows)*(size_cols)*sizeof(double));

    //performing matrix multiplication for each process:
    int i, j, k;
    double buff_sum;
    //the first 2 fors are for iterating over the elements of Cxy
    for(i=0; i<size_rows; i++){
      for(j=0; j<size_cols; j++){
        //the third for is for the inner product
        buff_sum = 0;
        for(k=0; k<m; k++){
          buff_sum += portion_A[i*m + k]*portion_B[j*m + k];
        }
        //and after the inner product, the Cxy element is assigned
        array3[i*size_cols + j] = buff_sum;
      }
    }

    //finally, send the resulting sub-matrix to the root process
    MPI_Send( array3, size_rows*size_cols, MPI_DOUBLE, root_process, 1, MPI_COMM_WORLD);
  }

  /* Stop this process */
  MPI_Finalize();
}



//IMPLEMENTATION OF FUNCTIONS:

//Function to check if string is int
int check_if_int(char str_buff[]){
  for(int i=0; i<strlen(str_buff); i++){
    if(!isdigit(str_buff[i])){return 0;}
  }
  return 1;
}

//Function for obtaining the best split dimensions for 2D grid
int best_divisor_q(int num_procs, int l, int n){

  if(num_procs == 2){
    if(l>n){return 2;}
    else{return 1;}
  }
  if(num_procs == 4){
    return 2;
  }

  int floor_sqrt;
  //floor of sqrt of num_procs
  floor_sqrt = (int)sqrt((double)num_procs);

  int buff_greatest_nr, buff_lowest_nr;
  //First, check which is the largest from l and n
  if(l>n){buff_greatest_nr = l; buff_lowest_nr = n;}
  else{buff_greatest_nr = n; buff_lowest_nr = l;}

  int i;
  //This is based in the divisors of num_procs
  for( i=floor_sqrt+1; i<num_procs; i++ ){
    if( num_procs%i==0 && buff_greatest_nr%i==0 ){
      if( buff_lowest_nr%(num_procs/i)==0 ){
        if( l>n ){return i;}
        else{return num_procs/i;}
      }
    }
  }
  //if not found, return -1
  return -1;
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
