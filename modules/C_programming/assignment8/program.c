#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 


//Function to find the position of the largest its in an array
int find_max(int *a, int n){
  //a is the array, n is total size (n*m in original notation)
  int i, max, index;

  max = a[0];
  index = 0;
  for(i=1; i<n; i++){
    if (a[i] > max) {
       index = i;
       max = a[i];
    }
  }
  return index;
}


//Function to find the min of three ints
int min_f(int a, int b, int c){
  if(a<=b && a<=c){return a;}
  else if(b<=a && b<=c){return b;}
  else{return c;}
}


//Function for printing the maze
void print_matrix(int *matrix_data, int n, int m){
  int i;
  for(i=0; i<n*m; i++){
    if(i%m == 0){printf("\n");}
    printf("%d", matrix_data[i]);
  }
  printf("\n\n");
}


//Main function
int main(){

  int n=0, m=0, i, j, index;
  char line[25];
  int *matrix_data, *matrix_aux_data;
  char *token;

  printf("\nProgram for finding maximal sub-matrix.\n\n");

  //Reading input file
  FILE *inp_file;
  inp_file = fopen("matrix.txt", "r");

  //..first line
  fgets(line, sizeof(line), inp_file);
  //Extracting info in first line of matrix.txt
  token = strtok(line, " ");
  n = atoi(token);
  token = strtok(NULL, " ");
  m = atoi(token);

  matrix_data = malloc(n*m*sizeof(int));
  matrix_aux_data = malloc(n*m*sizeof(int));
 
  //..reading the rest of the file line by line
  for(i=0; i<n; i++){
    fgets(line, sizeof(line), inp_file);
    token = strtok(line, " ");
    matrix_data[i*m+0] = atoi(token);
    for(j=1; j<m; j++){
      token = strtok(NULL, " ");
      matrix_data[i*m+j] = atoi(token);
    }
  }
  fclose(inp_file);
 
  printf("Input matrix:\n\n");
  print_matrix(matrix_data, n, m);

  //Constructing the aux matrix
  //..first row: the same
  for(i=0; i<m; i++){
    matrix_aux_data[i] = matrix_data[i];
  }

  //..first column: the same
  for(i=0; i<n; i++){
    matrix_aux_data[i*m] = matrix_data[i*m];
  }

  //..the other entries: determine if they represent the
  //  right-bottom corner of a maximal sub-matrix
  for(i=1; i<n; i++){
    for(j=1; j<m; j++){
      if(matrix_data[i*m+j] == 1){
        matrix_aux_data[i*m+j] = min_f(matrix_aux_data[(i-1)*m+j],
            matrix_aux_data[i*m+(j-1)],matrix_aux_data[(i-1)*m+(j-1)])+1;
      }
    }
  }

  //Printing the aux matrix
  printf("Aux matrix:\n");
  print_matrix(matrix_aux_data, n, m);

  //Finding the position of the max int in the aux matrix
  index = find_max(matrix_aux_data, n*m);

  //and printing the desired output to file (finding other ints
  //  with same value as variable 'index')
  FILE *out_file;
  out_file = fopen("result.txt", "w");
  //writing one of the max elements, to output
  fprintf(out_file, "%d\n", matrix_aux_data[index]);
  fprintf(out_file, "%d %d\n", (index-index%m)/m, index%m);
  //and in case that other elements in matrix_aux_data
  //  correspond to the max value too, then those are also
  //  solutions for the maximal sub-matrix
  for(i=1; i<n*m; i++){
    if(i==index){continue;}
    if(matrix_aux_data[i] == matrix_aux_data[index]){
      fprintf(out_file, "%d\n", matrix_aux_data[i]);
      fprintf(out_file, "%d %d\n", (i-i%m)/m, i%m);
    }
  }
  fclose(out_file);

  //Releasing memory
  free(matrix_data);
  free(matrix_aux_data);

  printf("\n");
  return 0;
}
