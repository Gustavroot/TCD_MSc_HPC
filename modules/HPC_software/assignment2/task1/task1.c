#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


//Compilation instructions:
//	gcc task1.c -lm

//Execution example:
//	./a.out -cs 70 -ps 25 -mr 1 -cr 75 -ng 100


//Summary on bitwise operators:
//	~: ones completement operator ('flipping bits')
//	&: corresponds to the bit-by-bit AND gate
//	^: XOR gate (1 if bit is set in one but not both)
//	|: OR gate


//Convention:
//**each individual has a chromosome structure: [int][int]...[int][int]
//**if, for example, the chromosome size is 70, then the structure
//  will be: [int 8][int 32][int 32], i.e. in the first integer, the
//  information of the first 24 bits is not taken into account
//In SUMMARY: the convention for bits, is that, for each array
//  of ints for each individual, for the first element of such array,
//  there is a certain # of bits not taken into account


void convert_int_to_bin(char* str_chrom, int n, int cutoff);
void print_int_as_bin(int n, int cutoff);
int fitness_fnctn(int n, int cutoff);
void print_population(int* pop_data, int size, int chrom_size);



//Main code
int main(int argc, char *argv[]){

  //random seed:
  srand((unsigned)time(NULL));

  int chrom_size, population_size, mutation_rate, crossover_rate, nr_generations;
  int i, j, indiv_size, buff_int, k;
  int *total_info;
  double buff_doub;

  printf("\nSerial code for the simple genetic algorithm (1s in a string).\n\n");

  if((argc-1)%2 != 0){
    printf("Wrong number of parameters.\n");
    return 0;
  }

  //int nr_iterations;
  //Reading command line arguments
  for(i=1; i<argc; i++){
    if(strcmp(argv[i], "-cs") == 0){
      chrom_size = atoi(argv[i+1]);
    }
    else if(strcmp(argv[i], "-ps") == 0){
      population_size = atoi(argv[i+1]);
    }
    else if(strcmp(argv[i], "-mr") == 0){
      mutation_rate = atoi(argv[i+1]);
    }
    else if(strcmp(argv[i], "-cr") == 0){
      crossover_rate = atoi(argv[i+1]);
    }
    else if(strcmp(argv[i], "-ng") == 0){
      nr_generations = atoi(argv[i+1]);
    }
    //else if(strcmp(argv[i], "-ni") == 0){
    //  nr_iterations = atoi(argv[i+1]);
    //}
    else{
      if(i%2 != 0){
        printf("Wrong format in data.. exiting now.\n\n");
        return 0;
      }
    }
  }

  if(i != 11){
    printf("Wrong format in data.. exiting now.\n\n");
    return 0;
  }

  //Listing specs for cache
  printf("Specs of system:\n");

  printf(" - chromosome size: %d\n", chrom_size);
  printf(" - population size: %d\n", population_size);
  printf(" - mutation rate (percentage): %d\n", mutation_rate); //this is a percentage
  printf(" - crossover rate: %d\n", crossover_rate); //this is a percentage
  printf(" - number of generations: %d\n", nr_generations);
  //printf(" - number of iterations: %d\n", nr_iterations);

  //Allocating memory for the genetic information of all the population
  //sizeof() gives the nr of bytes, and 1 byte = 8 bits
  //..number of ints necessary to store genetic info for each element
  //  of the population:
  j = 8*sizeof(int);
  i = (chrom_size-chrom_size%j)/j;
  if(chrom_size%j != 0){i++;}
  //..and then, total genetic info for the population:
  total_info = (int*)malloc(population_size*i*sizeof(int));
  //following variable stores the nr of ints used to specify one chromosome
  indiv_size = i;

  //Random initialisation of the population: generation of random ints
  //nr of bits in one int is sizeof(int)*8
  //In general, some irrelevant bits are set with the following loop
  for(i=0; i<population_size*indiv_size; i++){
    //max possible int
    buff_int = pow(2, sizeof(int)*8-1);
    //generate random ints with 9 digits:
    while(1){
      //numbers between 0 and 9999999999
      buff_doub = ((double)rand()/(double)RAND_MAX)*pow(10,10);
      //numbers between -9999999999/2 and 9999999999/2
      buff_doub = buff_doub-pow(10,10)/2;
      //now, restricting generated numbers to avoid overflow in ints
      if(buff_doub < (double)buff_int && buff_doub > (double)(-buff_int-1)){
        j = (int)buff_doub;
        break;
      }
    }
    //printf("%d\n", j);
    total_info[i] = j;
  }

  //DEBUG print
  //printf("%d\n", ~(127));
  //print_int_as_bin(127);

  //printing the state of the population
  //print_population(total_info, population_size*indiv_size, chrom_size);
  printf("\n\n");

  //TODO: create here array of floats to store fitnesses
  //TODO: create float file to store total fitness of population
  //TODO: open output file to store total fitnesses
  //TODO: create a buffer array, to store the offsprings; the size of
  //this array is population_size*chrom_size
  //TODO: create here an array for the mutation step, of size
  //population_size*chrom_size*(mutation_rate/100)

  //Iterating over all the generations
  for(i=0; i<nr_generations; i++){
    printf("\ngeneration #%d\n", i);
    print_population(total_info, population_size*indiv_size, chrom_size);
    printf("\n\n");

    //TODO: calculate the fitness of all the members, separately, and store it
    //in an array of floats

    //TODO: calculate the total fitness and store it in output file

    //TODO: iterate (population_size*crossover_rate)/2 times, and in each step
    //select 2 current chromosomes with wheel roulette selection, cross them,
    //create 2 new elements of offspring and store in buffer array for new generation

    //TODO: iterate [population_size-(population_size*crossover_rate)] times, and in
    //each step select a current chromosome with wheel roulette selection, and store
    //it in the buffer array for offspring info

    //TODO: iterate population_size*chrom_size*(mutation_rate/100) times, and in each
    //step flip one bit of the population_size*chrom_size bits of the offspring,
    //and also store the resulting index for that flipped bits, because no repetition
    //of indexes is permitted in this loop

/*
    //and for each generation, there is an iteration over individuals
    for(j=0; j<population_size; j++){
      //fitness:
      buff_int = 0;
      for(k=0; k<indiv_size; k++){
        if(k==0){buff_int += fitness_fnctn(total_info[indiv_size*j+k], indiv_size*(8*sizeof(int))-chrom_size);}
        else{buff_int += fitness_fnctn(total_info[indiv_size*j+k], 0);}}
      printf("%.4d -- %d\n", j, buff_int);
    }
*/

  //TODO: pass info in buffer array, to original array, to keep buffer array
  //available for the information of the next generation
  }
  

  //TODO: Release ALL memory
  free(total_info);
  printf("\n");

  return 0;
}


//return array of chars from given int
void convert_int_to_bin(char* str_chrom, int n, int cutoff){
  int buff_int = n, ctr=0, i, j;

  if(n<0){buff_int = ~n;}
  j = 8*sizeof(int)-cutoff;

  //binary representation, backwards
  while(buff_int>1){
    if(ctr==j){return;}
    if(buff_int%2 == 0){
      if(n<0){str_chrom[ctr] = '1';}
      else{str_chrom[ctr] = '0';}
    }
    else{
      if(n<0){str_chrom[ctr] = '0';}
      else{str_chrom[ctr] = '1';}
    }
    buff_int = buff_int/2;
    ctr++;
  }

  if(n<0){str_chrom[ctr] = '0';}
  else{str_chrom[ctr] = '1';}

  //for the sign bit:
  if(ctr==j){return;}
  ctr++;
  if(n<0){str_chrom[ctr] = '1';}
  else{str_chrom[ctr] = '0';}

  //filling the rest of info with zeros:
  buff_int = ctr;
  for(i=0; i<(8*sizeof(int))-(ctr)-1; i++){
    if(buff_int==j){return;}
    buff_int++;
    str_chrom[buff_int] = '0';
    //printf("0");
  }
}


//printing an int as binary
void print_int_as_bin(int n, int cutoff){
  //int i, ctr = 0, buff_int=n;
  int i;
  char* str_chrom;

  //m-allocation for binary info for printing:
  str_chrom = (char*)malloc((8*sizeof(int)-cutoff)*sizeof(char));

  convert_int_to_bin(str_chrom, n, cutoff);

  //and printing the forward representation
  for(i=(8*sizeof(int)-cutoff); i>=0; i--){
    printf("%c", str_chrom[i]);
  }

  free(str_chrom);
}


//Fitness function
//although mutation, crossover, etc, may be applied over all the bits,
//  fitness function applies exclusively over the number of bits
//  specified with the flag -cs
//This fitness function counts the # of 1s in the chromosome
int fitness_fnctn(int n, int cutoff){
  int i, j, k;
  char* str_chrom;

  //m-allocation for binary info for printing:
  k = 8*sizeof(int)-cutoff;
  str_chrom = (char*)malloc(k*sizeof(char));

  convert_int_to_bin(str_chrom, n, cutoff);
  j=0;
  for(i=0; i<k; i++){
    //parsing from char to int, by subtracting '0' (= 48)
    j += str_chrom[i]-48;
  }

  free(str_chrom);
  return j;
}


//Print the population data.. in binary representation
void print_population(int* pop_data, int size, int chrom_size){
  int i, cutoff, indiv_size;

  //'cutoff' indicates the bits to be unused
  i = 8*sizeof(int);
  i = (chrom_size-chrom_size%i)/i;
  if(chrom_size%(8*sizeof(int)) != 0){i++;}
  indiv_size = i;
  cutoff = i*(8*sizeof(int))-chrom_size;

  //printf("%.4d. -- ", 1);
  //in case the number is negative, the two's complement notation is used
  for(i=0; i<size; i++){
    if(i%indiv_size == 0){
      printf("\n");
      printf("%.4d. -- ", i/indiv_size);
      print_int_as_bin(pop_data[i], cutoff);
    }
    else{print_int_as_bin(pop_data[i], 0);}
    //printf("(%d)\n", cutoff);
  }
}
