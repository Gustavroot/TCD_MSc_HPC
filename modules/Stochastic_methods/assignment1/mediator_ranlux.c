#include <stdio.h>
#include <stdlib.h>
#include "ranlxs.h"

//On compiling: http://courses.cms.caltech.edu/cs11/material/c/mike/misc/compiling_c.html
//For compilation:
//	gcc -c -o ranlxs.o ranlxs.c
//	gcc -c -o mediator_ranlux.o mediator_ranlux.c
//	gcc -o mediator_ranlux mediator_ranlux.o ranlxs.o

//The following code generates a number argv[1] of
//random numbers, and prints them in an array in Python-like
//format
int main(int argc, char *argv[]){

  //First, convert the passed string to a int
  int nr_rands = strtol(argv[1], NULL, 0);

  float *rand_array = (float *)malloc(nr_rands*sizeof(float));
  ranlxs(rand_array, nr_rands);
  printf("[");

  int i;
  for(i=0; i<(nr_rands-1); i++){
    printf("%f,", rand_array[i]);
  }
  printf("%f", rand_array[nr_rands-1]);
  printf("]");

}
