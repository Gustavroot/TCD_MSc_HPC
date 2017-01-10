#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

//Compilation instructions:
//	$ gcc cache.c -lm

//Execution instructions:
//	$ ./cache -s 128 -l 16 -a 2 -f addressfile


//The cache simulated here, takes into account only 2 characteristics:
//(i.e. stores only 2 values per line)
//	- address in memory. If no data, this value is -1
//	- index of usage. By default, all lines have 0 as index of usage. The
//	idea is that with every miss, all of the indexes of usage in the set
//	are increased (except for those with -1 in memory address), and if no
//	-1 address available, then the line with the highest index_of_usage is
//	used to store the new memory
//	IMPORTANT: an index of usage of 0 means very frequently used


//Function to search for data in cache
//..for returning values, 1 is hit, 0 is miss
int query_cache(int **sim_cache_buff, int data_address, int lines_per_set, int set_id_buff){
  printf("%d\n", lines_per_set);
  //All the cache logic goes here:
  //..following for loop is for hits, and looks in corresponding set
  int min_i = set_id_buff*lines_per_set;
  int max_i = min_i + lines_per_set;
  int i, j;
  //this for loop restricts to search only in the corresponding set in cache
  for(i=min_i; i<max_i; i++){
    //0th entry is address
    if(sim_cache_buff[i][0] == data_address){
      //before returning a hit, the indexes of usage have to be increase:
      //(except those with -1 as address)
      for(j=min_i; j<max_i; j++){
        if(sim_cache_buff[j][1] != -1){sim_cache_buff[j][1]++;}
      }
      //and the currently read value is marked as the most recently used
      sim_cache_buff[i][1] = 0;
      printf("...it's a hit!\n");
      return 1;
    }
  }

  //If there was no success in finding the data in cache
  //..it is necessary to bring it from memory, and assign it to cache
  //..first, try to find available spaces in cache
  for(i=min_i; i<max_i; i++){
    //0th entry is address
    if(sim_cache_buff[i][0] == -1){
      for(j=min_i; j<max_i; j++){
        if(sim_cache_buff[j][1] != -1){sim_cache_buff[j][1]++;}
      }
      //and the currently read value is marked as the most recently used
      sim_cache_buff[i][1] = 0;
      sim_cache_buff[i][0] = data_address;
      printf("...it's a miss, but no cache line replacement was necessary.\n");
      return 0;
    }
  }
  //..if no cache line available, a replacement is necessary:
  //(but as all of them are being 'equally' increased, the index of usage 0 is taken)
  for(i=min_i; i<max_i; i++){
    if(sim_cache_buff[i][1] == 0){
      //have to increase the index of usage
      for(j=min_i; j<max_i; j++){
        sim_cache_buff[j][1]++;
      }
      //and the currently read value is marked as the most recently used
      sim_cache_buff[i][1] = 0;
      sim_cache_buff[i][0] = data_address;
      printf("...it's a miss, and cache line replacement was necessary.\n");
      return 0;
    }
  }
}


//Function to convert hex to decimal, returns int value
int from_hex_to_int(char hex_str[]){
  int hex_int;
  sscanf(hex_str, "%x", &hex_int);
  return hex_int;
}

//Function to convert hex str to bin str
char *from_hex_to_bin(char hex_str[], char *output_bin_str){
  char *bins_array[16] = {"0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111", "1000", "1001", "1010\n", "1011", "1100", "1101", "1110", "1111"};
  //char output_bin_str[12];
  //output_bin_str[0] = '\n';
  int i;
  for(i=0; i<4; i++){
    if(hex_str[i] == 'a'){strcpy(output_bin_str+i*4, bins_array[10]);}
    else if(hex_str[i] == 'b'){strcpy(output_bin_str+i*4, bins_array[11]);}
    else if(hex_str[i] == 'c'){strcpy(output_bin_str+i*4, bins_array[12]);}
    else if(hex_str[i] == 'd'){strcpy(output_bin_str+i*4, bins_array[13]);}
    else if(hex_str[i] == 'e'){strcpy(output_bin_str+i*4, bins_array[14]);}
    else if(hex_str[i] == 'f'){strcpy(output_bin_str+i*4, bins_array[15]);}
    else{strcpy(output_bin_str+i*4, bins_array[hex_str[i]-48]);}
  }
  //printf("%s", output_bin_str);
  return output_bin_str;
}


//Function to extract Set ID from address
int get_set_id(char hex_str[], double assoc_type){
  //Avoid using fourth element in string
  double nr_digits = log2(assoc_type);
  //..convertion from hex to bin
  char output_bin_str[16];
  from_hex_to_bin(hex_str, output_bin_str);

  //..and going from 12th element to the left, in bin string
  int i;
  int set_value = 0;
  for(i=11; i>(11-(int)nr_digits); --i){
    set_value += ((output_bin_str[i]-48))*pow(2, 11-i);
  }

  return set_value;
}



int main(int argc, char* argv[]){

  if((argc-1)%2 != 0){
    printf("Wrong number of parameters.\n");
    return 0;
  }

  int size_cache;
  int size_line_cache;
  int associative_type;
  char filename_addrss[100];
  //Reading command line arguments
  int i;
  for(i=1; i<argc; i++){
    if(strcmp(argv[i], "-s") == 0){
      size_cache = atoi(argv[i+1]);
    }
    else if(strcmp(argv[i], "-l") == 0){
      size_line_cache = atoi(argv[i+1]);
    }
    else if(strcmp(argv[i], "-a") == 0){
      associative_type = atoi(argv[i+1]);
    }
    else if(strcmp(argv[i], "-f") == 0){
      strcpy(filename_addrss, argv[i+1]);
    }
    else{
      if(i%2 != 0){
        printf("Wrong format in data.. exiting now.\n\n");
        return 0;
      }
    }
  }

  if(i != 8){
    printf("Wrong format in data.. exiting now.\n\n");
    return 0;
  }

  //Listing specs for cache
  printf("\nProgram for simulation of a cache.\n\n");
  printf("Specs of cache:\n");

  printf(" - size of cache: %d\n", size_cache);
  printf(" - size of one line: %d\n", size_line_cache);
  printf(" - set associative type: %d-way\n", associative_type);
  printf(" - filename for addresses: %s\n\n", filename_addrss);

  //Creating cache
  //..number of lines is size over line size:
  int cache_lines = size_cache/size_line_cache;
  int **sim_cache;
  //Each element in this array is a cache line
  sim_cache = malloc(cache_lines*sizeof(*sim_cache));
  for(i=0; i<cache_lines; i++){
    //one entry for the address of data in memory,
    //and a second entry for an index of usage (because of policy LRU)
    sim_cache[i] = malloc(2*sizeof(*sim_cache[i]));
  }

  //Initializing data in cache:
  //..addresses are all set to -1
  //..index of usage is set to 0 for all cache lines
  for(i=0; i<cache_lines; i++){
    //addresses
    sim_cache[i][0] = -1;
    //index of usage
    sim_cache[i][1] = 0;
  }

  printf("Testing cache (0 is miss, 1 is hit).\n");
  printf("------------------------------------\n\n");
  //Once the cache has been specified, addresses are read from file
  //and the cache is tested
  FILE* file = fopen(filename_addrss, "r");
  char line_address[10];
  int general_count_requests = 0;
  int count_hits = 0;
  int hit_or_miss;
  while(fgets(line_address, sizeof(line_address), file)){
    int set_id = get_set_id(line_address, (double)associative_type);
    printf("Searching for data in memory at address %s in cache set No. %d\n", line_address, set_id);
    hit_or_miss = query_cache(sim_cache, from_hex_to_int(line_address), cache_lines/associative_type, set_id);
    printf("Miss or hit: %d.\n", hit_or_miss);
    printf("\n");
    count_hits += hit_or_miss;
    general_count_requests++;
  }

  fclose(file);

  printf("\n------------\n");
  printf("hit rate = %f perc.\n", 100*((double)count_hits)/((double)general_count_requests));

  printf("\n");
  return 0;
}


