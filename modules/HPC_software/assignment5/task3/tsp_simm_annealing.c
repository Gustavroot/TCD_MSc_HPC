#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <getopt.h>
#include <ctype.h>


//COMPILATION instructions:
//	$ gcc -o tsp_simm_annealing tsp_simm_annealing.c -lm

//Example of execution for the att532.tsp file:
//	$ ./tsp_simm_annealing -n 532 -f att532.tsp

typedef struct{
  double x, y;
} city_point;

typedef struct{
  city_point* cities;
  double tour_length;
} solution_city;


//templates for functions
void compute_tour(solution_city*, int);
void perturb_tour(solution_city*, int);
int simulated_annealing(solution_city*, solution_city*, solution_city*, int, double, int, double);
void copy_solution(solution_city*, solution_city*);
double euclidean_distance(double, double, double, double);

void print_usage(){
  printf("USAGE: ./tsp_simm_annealing -n POSITIVE_INT [-f FILENAME]\n");
}

//main code
int main(int argc, char** argv){
  //general-purpose counters
  int i, j;
  int counter;

  //some parameters for the system
  int option, nr_cities = -1;
  //overal iterations -- not the real total
  int nr_iterations = 50000;
  double initial_temp = 100000000;
  //..rate at which temperature is decreased
  double alpha = 0.99;
  //number of simulations
  int nr_simulations = 3;
  
  printf("\nSIMULATED ANNEALING FOR TSP PROBLEM:\n\n");
  
  printf("\nSimulated annealing settings:\n");
  printf(" -- alpha (rate of decreasing T): %.5f\n", alpha);
  printf(" -- initial temperature: %.5f\n", initial_temp);
  printf(" -- number of overall simulations: %d\n", nr_simulations);
  printf(" -- number of iterations: %d\n", nr_iterations);
  
  printf("\n\n");

  if(argc != 3 && argc != 5){
    print_usage();
    return 0;
  }
  
  char filename[100];
  filename[0] = -1;

  //getting value of N with getopt
  while((option = getopt(argc, argv,"n:f:")) != -1){
    switch(option){
      case 'n':
        nr_cities = atoi(optarg);
        break;
      case 'f':
        strcpy(filename, optarg);
        break;
      default:
        print_usage();
        return 0;
    }
  }
  
  //check that N val is a positive integer
  if(nr_cities <= 0){
    printf("ERROR: number of cities must be > 0.\n");
    return 0;
  }
  
  //seeding random
  srand(time(NULL));

  //output solution and buffers for cities configurations
  solution_city solution_i;
  solution_city solution_buff;
  solution_city solution_f;
  
  //setting number of cities dynamically
  solution_i.cities = (city_point*)malloc(nr_cities*sizeof(city_point));
  solution_buff.cities = (city_point*)malloc(nr_cities*sizeof(city_point));
  solution_f.cities = (city_point*)malloc(nr_cities*sizeof(city_point));
  
  //data points
  if(filename[0] != -1){
    printf("file passed: %s\n\n", filename);

    FILE *fp = fopen(filename, "r");
    
    //check if file exists, and if not, exit
    if( fp == NULL ){
      print_usage();
      return 0;
    }
    
    int ch = 0;
    char line_buff[100];
    //buffer char pointers to extract data points info
    char *char_buff, *char_buff2;
    
    i = 0; j = 0;
    while(( ch = fgetc(fp) ) != EOF){
      //in case there are more lines than the specified value of N
      if(j/2 >= nr_cities){
        break;
      }
      line_buff[i] = ch;
      i++;
      if(ch == '\n'){
        line_buff[i] = '\0';
        i=0;
        //in case of the non-header entries of the file
        if(isdigit(line_buff[0])){
          //extraction of x,y values from file
          char_buff = strstr(line_buff, " ");
          char_buff2 = strstr(char_buff+1, " ");
          *char_buff2 = '\0';
          solution_i.cities[j/2].y = strtod(char_buff2+1, NULL);
          solution_i.cities[j/2].x = strtod(char_buff+1, NULL);
          
          //because of two values, x and y, per line
          j += 2;
        }
      }
    }
    fclose(fp);
    //in case N exceeds the number of lines in the data file
    if(nr_cities > j/2){
      nr_cities = j/2;
    }
  }
  else{
    //set initial solution proposed from random
    for(i=0; i < nr_cities; i++){
      solution_i.cities[i].x = ((double)1*rand()/(RAND_MAX));
      solution_i.cities[i].y = ((double)1*rand()/(RAND_MAX));
    }
  }
  
  //before starting, shuffle all cities 10*nr_cities times
  for(i = 0; i<10*nr_cities; i++){
    perturb_tour(&solution_i, nr_cities);
  }
  printf("\nShuffled cities!\n\n");
  
  //after setting values for data points, compute the first tour
  compute_tour(&solution_i, nr_cities);

  for(j = 1; j < nr_simulations+1; j++){
    counter = simulated_annealing(&solution_i, &solution_buff, &solution_f, nr_cities, alpha, nr_iterations, initial_temp);

    printf("Simulation # %d: ", j);
    printf("smallest trajectory = %f, but previously calculated length = %8.2f\n\n", solution_f.tour_length, solution_buff.tour_length);
  }
  
  return 0;
}



//CORE functions

void compute_tour(solution_city* sol, int nr_cities){
  int i;
  double tour_length = 0.0;
  for(i=0; i < nr_cities-1; i++){
    tour_length += euclidean_distance( 
                           sol->cities[i].x, sol->cities[i].y,
                           sol->cities[i+1].x, sol->cities[i+1].y);
  }
  tour_length += euclidean_distance( 
                           sol->cities[nr_cities-1].x, 
                           sol->cities[nr_cities-1].y,
                           sol->cities[0].x, sol->cities[0].y);
  sol->tour_length = tour_length;
}

void perturb_tour(solution_city* sol, int nr_cities){
  int p1, p2;
  double x, y;
  //generate different values for p1 and p2
  do{
    p1 = (int)((double)(nr_cities)*rand()/(RAND_MAX+1.0));
    p2 = (int)((double)(nr_cities)*rand()/(RAND_MAX+1.0));
  }while(p1 == p2);
  //set those new values
  x = sol->cities[p1].x;
  y = sol->cities[p1].y;
  //swap
  sol->cities[p1].x = sol->cities[p2].x;
  sol->cities[p1].y = sol->cities[p2].y;
  sol->cities[p2].x = x;
  sol->cities[p2].y = y;
  //compute tour for that new proposal
  compute_tour(sol, nr_cities);
}

int simulated_annealing(solution_city* solution_i, solution_city* solution_buff, solution_city* solution_f, int nr_cities, double alpha, int nr_iterations, double initial_temp){

  //TODO: add code to determine after how many steps
  //the solution converges
  int converg_counter = 0;

  double temperature = initial_temp, delta_e;
  solution_city temp_solution;
  int i;
  int counter = 0;
  //Copy solution_i to solution_buff and solution_f
  copy_solution(solution_buff, solution_i);
  copy_solution(solution_f, solution_buff);
  //loop over all values for Annealing temperature
  while(temperature > 0.001){
    counter++;
    //to check if case the solution remains unchanged
    converg_counter++;
    //copy solution_buff to temp_solution
    copy_solution(&temp_solution, solution_buff);
    //
    for(i = 0; i < nr_iterations; i++){
      // Compute one random neighbor of temp_solution
      perturb_tour(&temp_solution, nr_cities);
      delta_e = temp_solution.tour_length - (*solution_buff).tour_length;
      //if the 'Metropolis' step is to be accepted because of configuration
      if (delta_e < 0.0) {
	// temp_solution is better than solution_buff
	copy_solution(solution_buff, &temp_solution);
	if ((*solution_buff).tour_length < (*solution_f).tour_length){
	  copy_solution(solution_f, solution_buff);
	  converg_counter = 0;
	}
      }
      else{
	//if the 'Metropolis' step is to be accepted because of
	//the second chance based on flat random (this step avoids getting stuck in local max)
        if (exp((-delta_e/temperature)) > ((double)rand()/(double)RAND_MAX)){
	  copy_solution(solution_buff, &temp_solution);
        }
      }
      
      if(converg_counter > 100){
        printf("\n..the solution has been the same during 1000 iterations.. ");
        printf("this is iteration number: %d\n\n", counter);
        return counter;
      }
    }
    temperature *= alpha;
  }
  return counter;
}


//AUX functions

//set previous solution as new one
void copy_solution(solution_city* newSol, solution_city* oldSol){
  memcpy(newSol, oldSol, sizeof(solution_city));
}
double euclidean_distance(double x1, double y1, double x2, double y2){
  return sqrt(pow(fabs(x1-x2), 2) + pow(fabs(y1-y2), 2));
}
