#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>


//Compilation instructions:
//	$ gcc -o tsp-serial tsp-serial.c -lm


//EXTRA functions

void print_usage(){
  printf("USAGE: ./tsp-serial -n POSITIVE_INT\n");
}

//The following two functions are used for
//obtaining the whole set of permutations
void swap(int *x1,int *x2){
  int x=*x1;
  *x1=*x2;
  *x2=x;
}
void permutations(int *arr, int start, int end, double* ptr_distance, double* cities_info, int* best_path){
  int i=0;
  if(start == end){
    //printing path's order
    int k;
    double distance = 0, delt_x, delt_y;
    for(k=0; k<end; k++){
      //printf("%d ",arr[k]);
    }
    //printf("\n");

    //getting distance for that path
    for(k=0; k<(end-1); k++){
      delt_x = cities_info[2*arr[k]] - cities_info[2*arr[k+1]];
      delt_y = cities_info[2*arr[k]+1] - cities_info[2*arr[k+1]+1];
      distance += sqrt(pow(delt_x, 2) + pow(delt_y, 2));
    }
    //and finally, adding the final to the initial point distance
    delt_x = cities_info[2*arr[0]] - cities_info[2*arr[end-1]];
    delt_y = cities_info[2*arr[0]+1] - cities_info[2*arr[end-1]+1];
    distance += sqrt(pow(delt_x, 2) + pow(delt_y, 2));

    //printing distance for that path
    if(distance < *ptr_distance){
      *ptr_distance = distance;
      for(k=0; k<end; k++){
        best_path[k] = arr[k];
      }
    }
    //printf("%.4f, max = %.4f\n", distance, *ptr_distance);
  }
  else{
    for(i=start; i<end; i++){
      swap(arr+start, arr+i);
      permutations(arr, start+1, end, ptr_distance, cities_info, best_path);
      swap(arr+start, arr+i);
    }
  }
}


//main code
int main(int argc, char** argv){

  //general-purpose counters
  int i, j;

  int option, N = -1;

  if(argc != 3){
    print_usage();
    return 0;
  }

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
  
  //allocating array of size 2*N for information on cities
  double* cities_info = (double*)malloc(N*2*sizeof(double));
  
  //cities allocated in a 1x1-size map
  for(i = 0; i<2*N; i++){
    cities_info[i] = (double) rand()/RAND_MAX;
  }
  
  //DEBUG print
  for(i=0; i<N; i++){
    //printf("%.4f %.4f\n", cities_info[2*i], cities_info[2*i+1]);
  }

  printf("\nBrute force implementation of TSP (N = %d):\n\n", N);

  gettimeofday(&begin, NULL);

  //--------------------
  int* cities_labels = (int*)malloc(N*sizeof(int));
  for(i=0; i<N; i++){
    cities_labels[i] = i;
  }
  
  //Initially, set distance to max possible value,
  //in this case meaning travelling only through
  //the diagonal of the 1x1 square
  double distance = N*sqrt(2.0);
  double* ptr_distance = &distance;

  int* best_path = (int*)malloc(N*sizeof(int));

  permutations(cities_labels, 0, N, ptr_distance, cities_info, best_path);
  
  //print best path and distance
  printf("PATH = ");
  for(i=0; i<N; i++){
    printf("%d ", best_path[i]);
    printf("--> ");
  }
  printf("%d\n", best_path[0]);
  printf("Distance = %f\n", distance);
  //--------------------

  gettimeofday(&end, NULL);
  d_t = (end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec)/1000000.0);
  

  printf("Execution time: %f\n", d_t);

  //releasing memory
  free(cities_info);
  free(cities_labels);
  free(best_path);
  
  printf("\n");

  return 0;
}
