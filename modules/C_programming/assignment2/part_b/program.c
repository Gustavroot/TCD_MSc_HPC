#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

//Tested on Ubuntu 16.04 LTS

//Compilation instructions:
//	gcc program.c -o program

//Program use
//	./program

//Ordering function for the read string, using bubble sorting
//(copy of string, to avoid modifying the original string)
void bubble_sort(char buff_str_tmp[]){
  char *buff_str4 = (char *)malloc((strlen(buff_str_tmp)+1) * sizeof(char));
  //Copying the str
  strcpy(buff_str4, buff_str_tmp);
  double diff_time;
  //start timing here
  clock_t begin = clock();
  int n = strlen(buff_str4);
  int temp, i, step;
  for(step = 0; step < n-2; ++step)
    for(i = 0; i < n-step-2; ++i)
    {
      if(buff_str4[i] > buff_str4[i+1])
      {
        temp = buff_str4[i];
        buff_str4[i] = buff_str4[i+1];
        buff_str4[i+1] = temp;
      }
    }
  clock_t end = clock();
  //Putting a '>' at the end of the string
  //...first, is necessary to save the length of the string, as its '\n'
  //character is going to be changed of position now
  int tmp_str_length = strlen(buff_str4);
  buff_str4[tmp_str_length-1] = '>';
  buff_str4[tmp_str_length] = '\n';
  //stop timing here

  diff_time = (double)(end - begin) / CLOCKS_PER_SEC;

  //Outputting sorted string to file
  FILE *file_buff  = fopen("output1.txt",  "w");
  fwrite("<", 1, 1, file_buff);
  fwrite(buff_str4, strlen(buff_str4), 1, file_buff);
  fwrite("Time spent for bubblesort: ", 27, 1, file_buff);
  fprintf(file_buff, "%g", diff_time);
  fwrite("sec", 1, 1, file_buff);
  fwrite("\n", 1, 1, file_buff);
  fclose(file_buff);
}

void merge_arrays(char *A, char *L, int left_count, char *R, int right_count){
  int i,j,k;
  //Indexes for the 3 arrays used in the merging process
  i = 0; j = 0; k =0;
  while(i<left_count && j<right_count) {
    if(L[i] < R[j])
      A[k++] = L[i++];
    else
      A[k++] = R[j++];
  }
  while(i < left_count) A[k++] = L[i++];
  while(j < right_count) A[k++] = R[j++];
}

void sort_by_merge(char *A, int n){
  int mid, i;
  char *L, *R;
  //If the array has less than two, elements, do nothing
  if(n < 2) return;
  //Middle point (for splitting), rounded int
  mid = n/2;
  //Split into sub-arrays
  L = (char *)malloc(mid*sizeof(char)); 
  R = (char *)malloc((n- mid)*sizeof(char)); 
  //Left sub-array
  for(i = 0; i<mid; i++)
    L[i] = A[i];
  //Right sub-array
  for(i = mid; i<n; i++)
    R[i-mid] = A[i];
  //Recursive calls with the sub-arrays
  sort_by_merge(L, mid);
  sort_by_merge(R, n-mid);
  //Merge of sorted sub-arrays
  merge_arrays(A, L, mid, R, n-mid);
  //Release memory used as buffer
  free(L);
  free(R);
}

//BRIEF REVIEW ON 'merge sort'
//This is one of the fastest sorting algorithms
//..is based in divide and conquer, i.e. the original array is splitted
//  in sub-arrays, until arrays of length 1 are obtained, and then a
//  bottom-up process is taken, merging sorted sub-arrays
//..one of the biggest features is the merge part, where the first
//  elements of sub-arrays are compared, and then they're put in
//  a new array, and the process is continued until both arrays are
//  merged with each other
//..a very good explanation of this algorithm can be found in the
//  sumplementary material of the course CS50 of Harvard:
//  https://www.youtube.com/watch?v=EeQ8pwjQxTM

//Ordering function for the read string, using merge sorting
//(copy of string, to avoid modifying the original string)
void merge_sort(char buff_str_tmp[]){
  char *orig_str = (char *)malloc((strlen(buff_str_tmp)+2) * sizeof(char));
  //Copying the str
  strcpy(orig_str, buff_str_tmp);
  double diff_time;
  //start timing here
  clock_t begin = clock();

  sort_by_merge(orig_str, strlen(orig_str)-1);

  //stop timing here
  clock_t end = clock();

  diff_time = (double)(end - begin) / CLOCKS_PER_SEC;

  //Outputting sorted string to file
  FILE *file_buff  = fopen("output2.txt",  "w");
  fwrite("<", 1, 1, file_buff);
  //fwrite(orig_str, strlen(orig_str), 1, file_buff);
  //Printing orig_str by for loop
  int i;
  for(i=0; i<strlen(orig_str)-1; i++){
    fprintf(file_buff, "%c", orig_str[i]);
  }
  fprintf(file_buff, "%s", ">\n");
  fwrite("Time spent for merge sort: ", 27, 1, file_buff);
  fprintf(file_buff, "%g", diff_time);
  fwrite("sec", 1, 1, file_buff);
  fwrite("\n", 1, 1, file_buff);
  fclose(file_buff);
}

//Function to delete spaces within a string
//(pointer is used, as there's a need to modify
//the string so strip spaces)
void strip_spaces(char *buff_str3){
  int original_length = strlen(buff_str3);
  int number_stripped_spaces = 0;
  for(int i=0; i<original_length; i++){
    if(buff_str3[i]=='\n'){break;} //this restriction can be made lighter
    else if(buff_str3[i+number_stripped_spaces]==' '){number_stripped_spaces++;}
    buff_str3[i] = buff_str3[i+number_stripped_spaces];
  }
  //printf("%s", buff_str3);
}

//Function to sort the read str
//Variable buff_str2 is the str to sort here, and
//no length is needed, as there's only one '\n' character
//(no explicit pointer is used, to avoid modifying the original string)
void sorter_fctn(char buff_str2[]){
  //First, strip the spaces in string
  //strip_spaces(buff_str2); //uncomment this line if omitting spaces
  //After stripping, a reordering is necessary
  bubble_sort(buff_str2);
  merge_sort(buff_str2);
}


int main(){
  printf("\n");
  printf("Program to test timing for chars organization.\n\n");

  FILE *file_tmp = fopen("./inputfile.txt", "r");

  //Printing first line in file
  if (file_tmp != NULL) {
    //String to store the file's first line
    char buff_str[1000];
    while(fgets(buff_str, sizeof buff_str, file_tmp) != NULL)
    {
      //Preparing general setting here for reading the whole file
      //...but in case of reading only the first line from file,
      //just the following line applies
      break;
    }
    fclose(file_tmp);

    //This function sorts the original string by both the bubble
    //sort and the merge, and then writes the final results to
    //different output files
    sorter_fctn(buff_str);
    printf("...output written to files ./output1.txt and ./output2.txt");
  }
  else {
    //printing error (when reading file) message on stderr
    perror("Something went wrong.");
  }
  printf("\n\n");
}
