#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


//Compilation instructions:
//	gcc task2.c -lm

//Execution example (important: 'pdg' means prisoner's dilema games,
//which means the number of iterations of PD to play):
//	./a.out -pdg 12 -ps 20 -mr 5 -cr 80 -ng 100

//Summary on bitwise operators:
//	~: ones completement operator ('flipping bits')
//	&: corresponds to the bit-by-bit AND gate
//	^: XOR gate (1 if bit is set in one but not both)
//	|: OR gate


//Convention:
//**each individual has a chromosome structure: [int][int]...[int][int]
//**if, for example, the chromosome size is 70, then the structure
//  will be: [8 bits][32 bits][32 bits], i.e. in the first integer, the
//  information of the first 24 bits is not taken into account
//In SUMMARY: the convention for bits, is that, for each array
//  of ints for each individual, for the first element of such array,
//  there is a certain # of bits not taken into account


void convert_int_to_bin(char* str_chrom, int n, int cutoff);
void print_int_as_bin(int n, int cutoff);
//'iterns_pd' is the nr of iterations of PD within the fitness function
int fitness_fnctn(int indiv_size, int cutoff, int iterns_pd, int pop_size, int *total_info, int elem_index);
void print_population(int* pop_data, int size, int chrom_size, int *fitnesses);
void crossover_ints(int *crossover_buff, int x, int y, int n);



//Main code
int main(int argc, char *argv[]){

  //random seed:
  srand((unsigned)time(NULL));

  int chrom_size, population_size, mutation_rate, crossover_rate, nr_generations, iterns_pd;
  int i, j, indiv_size, buff_int, k, w, i_rand;
  int *total_info, *offspring;
  double buff_doub;

  //the size of the chromosomes is fixed: 16
  chrom_size = 16;

  printf("\nSerial code for the simple genetic algorithm (Prisoner's Dilema).\n\n");

  if((argc-1)%2 != 0){
    printf("Wrong number of parameters.\n");
    return 0;
  }

  //int nr_iterations;
  //Reading command line arguments
  for(i=1; i<argc; i++){
    if(strcmp(argv[i], "-pdg") == 0){
      iterns_pd = atoi(argv[i+1]);
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

  printf(" - number of plays of PD: %d\n", iterns_pd);
  printf(" - population size: %d\n", population_size);
  printf(" - mutation rate (percentage): %d\n", mutation_rate); //this is a percentage
  printf(" - crossover rate: %d\n", crossover_rate); //this is a percentage
  printf(" - number of generations: %d\n", nr_generations);

  //variables creation
  int *fitness_values = (int *)malloc(population_size*sizeof(int));
  int crossover_chrom1, crossover_chrom2, crossover_buff[2], mutation_chrom;
  double u_rand, buff_u;
  char *str_chrom;
  //m-allocation for binary info:
  str_chrom = (char*)malloc((8*sizeof(int))*sizeof(char));

  buff_doub = (((double)population_size*(double)crossover_rate)/100.0)/2.0;
  if((buff_doub-(int)buff_doub) != 0){
    printf("\nInput error: make (ps*cr/100)/2 be an integer.\n");
    return 0;
  }

  buff_doub = ((double)population_size*(double)mutation_rate)/100.0;
  if((buff_doub-(int)buff_doub) != 0){
    printf("\nInput error: make ps*mr/100 be an integer.\n");
    return 0;
  }

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
    total_info[i] = j;
  }

  printf("\n\n");

  //Creation of a file to store total fitness of population
  FILE *file_fitness;
  file_fitness = fopen("fitness_in_time.txt", "w");

  //Create a buffer array, to store the offsprings; the size of
  //this array is population_size*indiv_size
  offspring = (int*)malloc(population_size*indiv_size*sizeof(int));

  //Iterating over all the generations
  for(i=0; i<nr_generations; i++){

    //Calculate the fitness of all the members, separately, and store it
    //in an array of floats
    for(j=0; j<population_size; j++){
      //fitness:
      buff_int = 0;
      for(k=0; k<indiv_size; k++){
        if(k==0){
          buff_int += fitness_fnctn(indiv_size, indiv_size*(8*sizeof(int))-chrom_size,
		iterns_pd, population_size, total_info, indiv_size*j+k);
        }
        else{buff_int += fitness_fnctn(indiv_size, 0, iterns_pd, population_size, total_info, indiv_size*j+k);}
      }
      fitness_values[j] = buff_int;
    }

    //Calculate the total fitness and store it in an output file
    buff_int = 0;
    for(j=0; j<population_size; j++){
      buff_int += fitness_values[j];
    }
    fprintf(file_fitness, "%d\n", buff_int);

    //Iterate (population_size*crossover_rate)/2 times, and in each step
    //select 2 current chromosomes with wheel roulette selection, cross them,
    //create 2 new elements of offspring and store in buffer array for new generation

    for(j=0; j<(population_size*crossover_rate/100)/2; j++){

      //select 2 current chromosomes: use of wheel roulette selection

      u_rand = (double)rand()/(double)RAND_MAX;
      buff_u = 0.0;
      for(k=0; k<population_size; k++){
        if( buff_u < u_rand && u_rand < buff_u + (double)fitness_values[k]/(double)buff_int ){
          crossover_chrom1 = k; break;
        }
        else{buff_u += (double)fitness_values[k]/(double)buff_int;}
      }

      u_rand = (double)rand()/(double)RAND_MAX;
      buff_u = 0.0;
      for(k=0; k<population_size; k++){
        if( buff_u <= u_rand && u_rand <= buff_u + (double)fitness_values[k]/(double)buff_int ){
          crossover_chrom2 = k; break;
        }
        else{buff_u += (double)fitness_values[k]/(double)buff_int;}
      }

      //switching info from both chromosomes and copying that to the offspring
      for(k=0; k<indiv_size; k++){offspring[j*indiv_size*2+k] = total_info[crossover_chrom1*indiv_size+k];}
      for(k=0; k<indiv_size; k++){offspring[indiv_size+j*indiv_size*2+k] = total_info[crossover_chrom2*indiv_size+k];}
      //Now, the two chromosomes are crossed (the 1st bit of both is interchanged)
      //Crossing the chromosomes means that, from the 2 n-plets of ints, the 1st ints are crossed,
      //in a way that the [8*sizeof(int)-(indiv_size*(8*sizeof(int))-chrom_size)]-th
      //bits in those two ints are interchanged
      crossover_ints(crossover_buff, total_info[crossover_chrom1*indiv_size],
		total_info[crossover_chrom2*indiv_size], (indiv_size*(8*sizeof(int))-chrom_size));
      offspring[j*indiv_size*2] = crossover_buff[0];
      offspring[indiv_size+j*indiv_size*2] = crossover_buff[1];
    }

    //Iterate [population_size-(population_size*crossover_rate)] times, and in
    //each step select a current chromosome with wheel roulette selection, and store
    //it in the buffer array for offspring info

    printf("\n");
    for(k=0; k<(population_size-population_size*crossover_rate/100); k++){
      //select 1 current chromosome: use of wheel roulette selection
      u_rand = (double)rand()/(double)RAND_MAX;
      buff_u = 0.0;
      for(w=0; w<population_size; w++){
        if( buff_u < u_rand && u_rand < buff_u + (double)fitness_values[w]/(double)buff_int ){
          mutation_chrom = w; break;
        }
        else{buff_u += (double)fitness_values[w]/(double)buff_int;}
      }
      for(w=0; w<indiv_size; w++){
        offspring[j*indiv_size*2+k*indiv_size+w] = total_info[mutation_chrom*indiv_size+w];
      }
    }

    //Iterate population_size*chrom_size*(mutation_rate/100) times, and in each
    //step flip one bit of the population_size*chrom_size bits of the offspring,
    //and also store the resulting index for that flipped bits, because no repetition
    //of indexes is permitted in this loop
    for(k=0; k<population_size*mutation_rate/100; k++){
      //to mutate, obtain a random integer from 0 to population_size*indiv_size
      i_rand = rand()%(population_size*indiv_size);
      //convert the selected int to bits
      convert_int_to_bin(str_chrom, offspring[i_rand], 0);
      w = i_rand;

      //from that subset of bits of the chromosome, only mutate over the relevant
      if(w%(indiv_size)==0){
        i_rand = rand()%(8*sizeof(int)-(indiv_size*(8*sizeof(int))-chrom_size));

        if(str_chrom[indiv_size*(8*sizeof(int))-chrom_size + i_rand] == 0){
          str_chrom[indiv_size*(8*sizeof(int))-chrom_size + i_rand] = 1;
        }
        else{
          str_chrom[indiv_size*(8*sizeof(int))-chrom_size + i_rand] = 0;
        }
      }
      else{
        i_rand = rand()%(8*sizeof(int));

        if(str_chrom[i_rand] == 0){
          str_chrom[i_rand] = 1;
        }
        else{
          str_chrom[i_rand] = 0;
        }
      }

      //and converting back to integer
      i_rand = w;
      offspring[i_rand] = 0;
      for(w=0; w<(8*sizeof(int)); w++){
        offspring[i_rand] += str_chrom[w]*pow(2,w);
      }

    }

    //DEBUG print:
    printf("\n\n------\n%d\n\n", i);
    printf("original gen:");
    print_population(total_info, population_size*indiv_size, chrom_size, fitness_values);
    printf("\n\noffspring (fitness values are not the corresponding!):");
    print_population(offspring, population_size*indiv_size, chrom_size, fitness_values);

    //Pass info in offspring[], to total_info[], to keep offspring[]
    //available for the information of the next generation
    for(j=0; j<population_size*indiv_size; j++){
      total_info[j] = offspring[j];
    }
  }
  
  fclose(file_fitness);

  //Release ALL memory
  free(total_info);
  free(fitness_values);
  free(str_chrom);
  free(offspring);
  printf("\n");

  return 0;
}



//Check this crossover, as might be reversing the treatment of bits within ints........
//Function to perform the crossover: given x and y, puts the nth
//  bit in y into x, and returns x
void crossover_ints(int *crossover_buff, int x, int y, int cutoff){

  int i, j;
  char *str_chrom_x, *str_chrom_y;

  //m-allocation for binary info for printing:
  str_chrom_x = (char*)malloc((8*sizeof(int)-cutoff)*sizeof(char));
  str_chrom_y = (char*)malloc((8*sizeof(int)-cutoff)*sizeof(char));

  convert_int_to_bin(str_chrom_x, x, cutoff);
  convert_int_to_bin(str_chrom_y, y, cutoff);

  //the crossover step:
  i = str_chrom_x[8*sizeof(int)-cutoff-1];
  str_chrom_x[8*sizeof(int)-cutoff-1] = str_chrom_y[8*sizeof(int)-cutoff-1];
  str_chrom_y[8*sizeof(int)-cutoff-1] = i;

  //and converting back those bits to integers
  j = 0;
  for(i=(8*sizeof(int)-cutoff-1); i>=0; i--){
    j += (str_chrom_x[i]-48)*pow(2, i);
  }
  crossover_buff[0] = j;//+pow(2, 8*sizeof(int)-cutoff); //add an extra 1
  j = 0;
  for(i=(8*sizeof(int)-cutoff-1); i>=0; i--){
    j += (str_chrom_y[i]-48)*pow(2, i);
  }
  crossover_buff[1] = j;//+pow(2, 8*sizeof(int)-cutoff); //add an extra 1

  free(str_chrom_x);
  free(str_chrom_y);
}

//return array of chars from given int
//IMPORTANT: this returns the bin representation in reversed order,
//e.g.: integer 4 is 100, and the returned array using this
//function would be: ['0', '0', '1']
void convert_int_to_bin(char* str_chrom, int n, int cutoff){
  int buff_int = n, ctr=0, i, j;

  if(n<0){buff_int = ~n;}
  j = 8*sizeof(int)-cutoff;

  if(n==0){
    for(i=0; i<j; i++){
      str_chrom[i] = '0';
    }
    return;
  }

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


//Print the population data.. in binary representation
void print_population(int* pop_data, int size, int chrom_size, int *fitnesses){
  int i, cutoff, indiv_size;

  //'cutoff' indicates the bits to be unused
  i = 8*sizeof(int);
  i = (chrom_size-chrom_size%i)/i;
  if(chrom_size%(8*sizeof(int)) != 0){i++;}
  indiv_size = i;
  cutoff = i*(8*sizeof(int))-chrom_size;

  //in case the number is negative, the two's complement notation is used
  for(i=0; i<size; i++){
    if(i%indiv_size == 0){
      printf("\n");
      printf("%.4d. -- (%.5d): ", i/indiv_size, fitnesses[i/indiv_size]);
      print_int_as_bin(pop_data[i], cutoff);
    }
    else{print_int_as_bin(pop_data[i], 0);}
  }
}

//Fitness function
//although mutation, crossover, etc, may be applied over all the bits,
//  fitness function applies exclusively over the number of bits
//  specified with the flag -cs
//This fitness function plays Prisoner's Dilema
//'elem_index' is the position of the current chromosome, in the array 'total_info'
int fitness_fnctn(int indiv_size, int cutoff, int iterns_pd, int pop_size, int *total_info, int elem_index){
  int i, j, k;
  char *str_chrom, *str_chrom_alice;
  int prev_game[2], prev_prev_game[2], gains[4], buff_game[2];
  int fitness_val;
  double buff_doub;
  //m-allocation for binary info for printing:
  k = 8*sizeof(int)-cutoff;
  str_chrom = (char*)malloc(k*sizeof(char));
  str_chrom_alice = (char*)malloc(k*sizeof(char));

  //To play PD, as it's based in the previous 2 games, the first two
  //games are generated.. randomly here
  buff_doub = (double)rand()/(double)RAND_MAX;
  if(buff_doub<0.5){prev_game[0] = 0;}
  else{prev_game[0] = 1;}
  buff_doub = (double)rand()/(double)RAND_MAX;
  if(buff_doub<0.5){prev_game[1] = 0;}
  else{prev_game[1] = 1;}
  buff_doub = (double)rand()/(double)RAND_MAX;
  if(buff_doub<0.5){prev_prev_game[0] = 0;}
  else{prev_prev_game[0] = 1;}
  buff_doub = (double)rand()/(double)RAND_MAX;
  if(buff_doub<0.5){prev_prev_game[1] = 0;}
  else{prev_prev_game[1] = 1;}

  //obtaining the bin representation of the chromosome (originally in int form)
  convert_int_to_bin(str_chrom, total_info[elem_index], cutoff);

  //creating the array of weights:
  //0='CC', 1='CD', 2='DC', 3='DD'
  gains[0] = 3;
  gains[1] = 5;
  gains[2] = 0;
  gains[3] = 1;
  //as can be seen from gains[], the chromosome for which the
  //fitness function is being obtained, is always playing as 'Bob'

  //following variable stores the fitness
  fitness_val = 0;

  //playing a round-robin tournament means playing once against
  //each one of the other chromosomes
  for(j=0; j<pop_size; j++){
    //avoid this chromosome playing agains itself
    if(j==elem_index){continue;}
    //playing PD 'iterns_pd' times against each other chromosome
    for(i=0; i<iterns_pd; i++){
      //'C' (co-operate) has an index value of 0, e.g. if the previous game
      //was [A,B] = [0,1] and the previous to that [A,B] = 
      //[1,0], then index = (0*2^3 + 1*2^2)+(1*2^1 + 0*2^0) = 6

      //based in the first two games, obtain the new outcome and
      //store it in 'buff_game' array, 0 for Alice and 1 for Bob:
      //game outcome for Bob (current chromosome)
      buff_game[1] = str_chrom[prev_game[0]*2^3 + prev_game[1]*2^2 + prev_prev_game[0]*2^1 +
		prev_prev_game[1]*2^0];

      //game outcome for Alice
      convert_int_to_bin(str_chrom_alice, total_info[i], cutoff);
      buff_game[0] = str_chrom_alice[prev_game[0]*2^3 + prev_game[1]*2^2 + prev_prev_game[0]*2^1 +
		prev_prev_game[1]*2^0];

      //based on those two results, the fitness is increased or not for this
      //iteration of the tournament
      fitness_val += gains[buff_game[0]*2^1 + buff_game[1]*2^0];

      //for the next match, switch results
      prev_prev_game[0] = prev_game[0];
      prev_prev_game[1] = prev_game[1];
      prev_game[0] = buff_game[0];
      prev_game[1] = buff_game[1];
    }
  }

  free(str_chrom);
  free(str_chrom_alice);
  printf("%d\n", fitness_val);
  return fitness_val;
}
