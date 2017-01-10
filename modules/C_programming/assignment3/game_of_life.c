#include <stdio.h>
#include <stdlib.h>


//Function to copy one matrix into another
void copy_matrices(int **matrix_A, int **matrix_B, int matrix_dim){
  //matrix_B is copied into matrix_A
  int i, j;
  for(i=0; i<matrix_dim; i++){
    for(j=0; j<matrix_dim; j++){
      matrix_A[i][j] = matrix_B[i][j];
    }
  }
}


//Function to re-set the boundaries (cyclic conditions)
void reset_boundaries(int **matrix_2d, int matrix_dim){
  int i;
  //...setting values of right wall
  for(i=1; i<(matrix_dim-1); i++){
    matrix_2d[i][matrix_dim-1] = matrix_2d[i][1];
  }
  //...setting values of left wall
  for(i=1; i<(matrix_dim-1); i++){
    matrix_2d[i][0] = matrix_2d[i][matrix_dim-2];
  }
  //...setting values of up wall
  for(i=1; i<(matrix_dim-1); i++){
    matrix_2d[0][i] = matrix_2d[matrix_dim-2][i];
  }
  //...setting values of down wall
  for(i=1; i<(matrix_dim-1); i++){
    matrix_2d[matrix_dim-1][i] = matrix_2d[1][i];
  }
  //..and now the corners
  matrix_2d[0][0] = matrix_2d[matrix_dim-2][matrix_dim-2];
  matrix_2d[matrix_dim-1][matrix_dim-1] = matrix_2d[1][1];
  matrix_2d[matrix_dim-1][0] = matrix_2d[1][matrix_dim-2];
  matrix_2d[0][matrix_dim-1] = matrix_2d[matrix_dim-2][1];
}


//Function for 1 iteration of the game of life
void one_step_evolution(int **M, int **M_buff, int dim_matrix){
  //First, update inner layers
  int i, j, count_neighbours;
  for(i=1; i<(dim_matrix-1); i++){
    for(j=1; j<(dim_matrix-1); j++){
      //..first, sum over states of nearest neighbours
      count_neighbours = 0;
      count_neighbours += M[i-1][j];
      count_neighbours += M[i+1][j];
      count_neighbours += M[i][j-1];
      count_neighbours += M[i][j+1];
      count_neighbours += M[i+1][j+1];
      count_neighbours += M[i-1][j-1];
      count_neighbours += M[i+1][j-1];
      count_neighbours += M[i-1][j+1];
      //..now, conditions on being alive or dead
      if(M[i][j] == 0 && count_neighbours == 3){M_buff[i][j] = 1;}
      else if(M[i][j] == 1 && (count_neighbours == 2 || count_neighbours == 3)){M_buff[i][j] = 1;}
      else{M_buff[i][j] = 0;}
    }
  }
  //then, re-set boundaries in buffer matrix
  reset_boundaries(M_buff, dim_matrix);
  //finally, copy full matrix to original
  copy_matrices(M, M_buff, dim_matrix);
}


//Function to print matrices
void matrix_print(int **M, int n_orig, int m_orig){
  //First, restrict printing a matrix dimension greater than 15
  int n, m;
  //if(n_orig>15){n=15;}
  //else{n = n_orig;}
  //if(m_orig>15){m=15;}
  //else{m = m_orig;}
  n = n_orig;
  m = m_orig;
  //Then, printing elements of matrix with double for loop
  for(int i=1; i<n-1; i++){
    for(int j=1; j<m-1; j++){
      printf("%d ", M[i][j]);
    }
    //if(m_orig>15){printf("...\n");}
    //else{printf("\n");}
    printf("\n");
  }
  //if(n_orig>15){printf("... \n");}
  printf("\n");
}


int main(){
  printf("\nProgram to execute the Game of Life over a fixed input file.\n");

  //First, read data from file and store in matrix
  //...file pointer and checking if file is not corrupted
  FILE *fptr;
  fptr = fopen("start_config.txt", "r");
  if(fptr == NULL){
    printf("Error reading initial state info file!");
    exit(1);
  }
  //...creating the 2D matrix
  char string_data[102];
  fgets(string_data, 102, fptr);
  int matrix_dim = atoi(string_data)+2;
  printf("Matrix dim: %d\n\n", matrix_dim-2);
  int **matrix_2d;
  matrix_2d = (int**)malloc(matrix_dim * sizeof(int*));
  int i;
  for(i=0; i<matrix_dim; i++){
    matrix_2d[i] = (int*)malloc(matrix_dim * sizeof(int));
  }
  //...reading the file and putting data into matrix
  int j, tmp_int;
  for(j=1; j<matrix_dim-1; j++){
    fgets(string_data, 102, fptr);
    //setting initial matrix to matrix_2d, without changing
    //the outter layer of 0s
    for(i=0; i<matrix_dim*2-4; i += 2){
      matrix_2d[j][i/2+1] = string_data[i]-'0';
    }
  }
  fclose(fptr);
  //Debugging: print initial matrix
  //matrix_print(matrix_2d, matrix_dim, matrix_dim);

  //Second, redefine that matrix to make it easier the take into
  //account the boundary conditions
  reset_boundaries(matrix_2d, matrix_dim);
  //Debugging: print initial matrix
  //matrix_print(matrix_2d, matrix_dim, matrix_dim);

  //Then, iterate 100 times over the expanded matrix to evolve
  //it in time
  //To avoid creating and destroying a matrix on each iteration,
  //just one matrix is used 'globally' as buffer
  int **buff_matrix_2d;
  buff_matrix_2d = (int**)malloc(matrix_dim * sizeof(int*));
  for(i=0; i<matrix_dim; i++){
    buff_matrix_2d[i] = (int*)malloc(matrix_dim * sizeof(int));
  }
  copy_matrices(buff_matrix_2d, matrix_2d, matrix_dim);
  //Debugging:
  printf("Initial state:\n");
  matrix_print(matrix_2d, matrix_dim, matrix_dim);
  //matrix_print(buff_matrix_2d, matrix_dim, matrix_dim);

  //for loop with 100 iterations
  for(i=0; i<100; i++){
    one_step_evolution(matrix_2d, buff_matrix_2d, matrix_dim);
  }
  //Debugging:
  printf("Final state:\n");
  //matrix_print(matrix_2d, matrix_dim, matrix_dim);
  matrix_print(matrix_2d, matrix_dim, matrix_dim);

  //Finally, store the reduced/inner matrix in an output file:
  FILE *fptr2;
  fptr2 = fopen("config100.txt", "w");
  fprintf(fptr2, "%d\n", matrix_dim-2);
  for(i=1; i<(matrix_dim-1); i++){
    for(j=1; j<(matrix_dim-1); j++){
      fprintf(fptr2, "%d ", matrix_2d[i][j]);
    }
    fprintf(fptr2, "\n");
  }
  fclose(fptr2);
  printf("\n");
}
