#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//NOTE 1: on structures, -> and . can be used to access members. On
//one hand, -> acts over pointers and . over structures. The use of
//the -> dereferencing operator is prefered here

//NOTE 2: one slight modification was made to the specifications of the
//problem. The extraction of matrix data through t.entry[x][y] is possible,
//but a mapping is necessary from the struct data to the real form of
//the tri-diagonal matrix. For that conversion/mappin, the function
//re_map_matr_entries(t_matrix, i, j) was created

//Definition of the structure to store the tri-diagonal matrix
struct Tridiag_s{
  int n_dim;
  double matrix_first_entry;
  double **entry;
};
//Defining a new type of variable (struct)
typedef struct Tridiag_s Tridiag;


//Function to map entries of tri-diag matrix to usual matrix entries
//..i is for rows, j for columns
double re_map_matr_entries(Tridiag *t, int i, int j){
  //double correct_entry_val;

  //Code to map entries the right way:
  if(i==0 && j==0){return t->matrix_first_entry;}
  else if(abs(i-j)>1){return 0;}
  else{
    if(i>j){return t->entry[2][j];}
    else{return t->entry[1-(j-i)][j-1];}
  }
};


//Function to print tri-diag matrix struct
void print_trid_matr(Tridiag *t){
  int i,j;
  for(i=0; i<t->n_dim; i++){
    for(j=0; j<t->n_dim; j++){
      printf("%.3f\t", re_map_matr_entries(t, i, j));
    }
    printf("\n");
  }
}


//Function to fill tri-diagonal matrix struct
//..takes file named filename_s, tri-diag struct
void tridiag_fill(Tridiag *t, char *filename_s){
  FILE *file = fopen(filename_s, "r");
  char line [1000];
  int i, j;//, matrix_dim;
  //Getting matrix dimension
  fgets(line, sizeof line, file);
  //matrix_dim = atoi(line);
  //Getting matrix data
  fgets(line, sizeof line, file);

  //Splitting matrix info from file and putting it into matrix_info array:
  double *matrix_info = malloc((t->n_dim+2*(t->n_dim-1))*sizeof(double));
  int counter=0;
  char *pch;
  pch = strtok(line, " ");
  while (pch != NULL)
  {
    matrix_info[counter] = strtod(pch, NULL); //atoi(pch);
    //printf("%s\n", pch);
    pch = strtok(NULL, " ");
    counter++;
  }
  //And assigning now info to members of the t struct
  t->matrix_first_entry = matrix_info[0];
  //printf("%d\n", ((t->n_dim+2*(t->n_dim-1))/3+1));
  //..before storing the other elements, it's necessary to mallocate
  //..looping in sub-groups of 3
  for(i=1; i<((t->n_dim+2*(t->n_dim-1))/3+1); i++){
    t->entry[0][i-1] = matrix_info[(i-1)*3+1];
    t->entry[1][i-1] = matrix_info[(i-1)*3+3];
    t->entry[2][i-1] = matrix_info[(i-1)*3+2];
  }
  fclose(file);
}

//Function for applying a tri-diagonal matrix to a vector
void tridiag_mul(Tridiag *t, double *a, double *b){
  int i, j;
  //Cleaning b, just in case
  for(i=0; i<t->n_dim; i++){
    b[i] = 0;
  }
  //Assigning the results to b
  for(i=0; i<t->n_dim; i++){
    for(j=0; j<t->n_dim; j++){
      b[i] += re_map_matr_entries(t, i, j)*a[j];
    }
  }
}


//Functions for inverting a tridiagonal matrix
//..internal code is added after main()
void trid_matrix_inverse(Tridiag *t);
double phi(Tridiag *t, int i, int n);
double theta(Tridiag *t, int i, int n);


//Main code:
void main(){
  printf("\nProgram for storing and using a matrix within a C struct:\n\n");

  //Instance of a new tri-diag matrix
  //Tridiag *trid_matrx;

  //For future calculations and allocations, the size of the matrix
  //is necessary at this point
  char filename_ext[] = "tridiag_matrix.txt";
  char matrix_dim[5];
  FILE *file = fopen(filename_ext, "r");
  fgets(matrix_dim, sizeof matrix_dim, file);
  fclose(file);
  int matrix_dim_sq = atoi(matrix_dim);

  //and allocating space in memory for that matrix struct
  int tridiag_elemts = matrix_dim_sq+2*(matrix_dim_sq-1);

  //..first, initializing the struct
  Tridiag trid_matrx_buff = {matrix_dim_sq, 0, malloc(((tridiag_elemts-1)/3)*sizeof(double*))};
  Tridiag *trid_matrx;
  trid_matrx = &trid_matrx_buff;
  //..then, mallocating for the sub-arrays
  int i;
  for(i=0; i<((tridiag_elemts-1)/3); i++){
    (trid_matrx->entry)[i] = malloc(3*sizeof(double));
  }

  //Next step is to fill the matrix with data from an external file
  //..and once filled, the struct has the matrix info
  tridiag_fill(trid_matrx, filename_ext);

  //Before proceeding to calculations, printing matrix for debug
  printf("Original tri-diag matrix:\n");
  print_trid_matr(trid_matrx);
  printf("\n");

  //Applying the matrix over a vector
  //..before calling the application matrix application function,
  //  the vector is created
  double *vect_a = malloc(trid_matrx->n_dim*sizeof(double));
  //..initializing this vector to the specific data given in the
  //  problem description
  //..in case of matrix-size extension, just add or delete elements
  //  from the following lines, but if not, execution won't crash
  vect_a[0] = 2.3;
  vect_a[1] = -4.4;
  vect_a[2] = 7.9;
  vect_a[3] = -0.1;
  vect_a[4] = 5.9;
  //..and for the resulting vector:
  double *vect_b = malloc(trid_matrx->n_dim*sizeof(double));

  //And now, matrix application is taken
  tridiag_mul(trid_matrx, vect_a, vect_b);

  printf("Vector resulting from multiplication:\n");
  //Printing the resulting vector vect_b
  for(i=0; i<trid_matrx->n_dim; i++){
    printf("%.3f\n", vect_b[i]);
  }
  //printf("\n");

  //Finally, inverting the tri-diagonal matrix
  //..and automatically saving the result to a file inverse.txt
  trid_matrix_inverse(trid_matrx);

  printf("\n");
}


//The following functions are used for the inversion process of the
//tridiagonal matrix

//..phi function used in the recursive inversion of the tridiag matrix
double phi(Tridiag *t, int i, int n)
{
  double p;
  if(i == n){return 1;}
  else if(i == n-1){return re_map_matr_entries(t, i, i);}
  else{p=re_map_matr_entries(t, i, i)*phi(t, i+1,n)-re_map_matr_entries(t, i, i+1)*re_map_matr_entries(t, i+1, i)*phi(t, i+2,n);}
  return p;
}

//..theta function used in the recursive inversion of the tridiag matrix
double theta(Tridiag *t, int i, int n)
{
  double p;
  if(i==-1){return 1;}
  else if(i==0){return re_map_matr_entries(t, i, i);}
  else{p=re_map_matr_entries(t, i, i)*theta(t, i-1,n)-re_map_matr_entries(t, i-1, i)*re_map_matr_entries(t, i, i-1)*theta(t, i-2,n);}
  return p;
}

//Mainly based on the general form explained at:
//  http://www.mat.uc.pt/preprints/ps/p0516.pdf
void trid_matrix_inverse(Tridiag *t){

  int n = t->n_dim;
  int i, j, a, b;
  //m-allocating the matrix T for storing resulting inverse
  double **T = malloc(n*sizeof(double*));  
  for(i=0; i<n; i++){
    T[i] = malloc(n*sizeof(double));
  }
  //Implementation of the core part of the inversion algorithm
  for(i=0; i<n;i++){
    for(j=0; j<n;j++){
      b=j;
      a=i;
      if(i<j){
        T[i][j]=pow(-1, i+j);
        for(a; a<b; a++){
          T[i][j]*=re_map_matr_entries(t, a, a+1);
        }
        T[i][j]*=theta(t, i-1,n)*phi(t, j+1,n)/theta(t, n-1,n);
      }
      else if(i==j){
        if(i==0){
          T[i][j]=0;
        }
        else{
          T[i][j]=theta(t, i-1,n)*phi(t, j+1,n)/theta(t, n-1,n);
        }
      }
      else{
        T[i][j]=pow(-1, i+j);
        for(b; b<i; b++){
          T[i][j]*=re_map_matr_entries(t, a, b);;
        }
        T[i][j]*=theta(t, j-1, n)*phi(t, i+1,n)/theta(t, n-1, n);
      }
    }
  }

  //And saving resulting matrix to file inverse.txt
  FILE* file_output  = fopen("inverse.txt",  "w");
  char char_buff[10];
  sprintf(char_buff, "%d", t->n_dim);
  fprintf(file_output, "%s\n", char_buff);
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      sprintf(char_buff, "%.3f\t", T[i][j]);
      fprintf(file_output, "%s", char_buff);
    }
    fprintf(file_output, "\n");
  }
  fclose(file_output);

  //But also displayed in the terminal
  printf("\nAfter matrix inversion:\n");
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      printf("%.3f\t", T[i][j]);
    }
    printf("\n");
  }
};
