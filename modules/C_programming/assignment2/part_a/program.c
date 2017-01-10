#include <stdio.h>
#include <string.h>

//Tested on Ubuntu 16.04 LTS

//Compilation instructions:
//	gcc program.c -o program

//Program use
//	./program

//Ordering function for the read string, using bubble sorting
//(no pointer used, to avoid modifying the original string)
void bubble_sort(char buff_str4[]){
  int n = strlen(buff_str4);
  int temp, i, step;
  for(step = 0; step < n-1; ++step)
    for(i = 0; i < n-step-1; ++i)
    {
      if(buff_str4[i] > buff_str4[i+1])
      {
        temp = buff_str4[i];
        buff_str4[i] = buff_str4[i+1];
        buff_str4[i+1] = temp;
      }
    }
  //Outputting sorted string to file
  FILE *file_buff  = fopen("outfile.txt",  "w");
  fwrite(buff_str4, strlen(buff_str4), 1, file_buff);
  fwrite("\n", 1, 1, file_buff);
  fclose(file_buff);
}

//Ordering function for the read string, using quick sorting
//(no pointer used, to avoid modifying the original string)
void quick_sort(char *buff_str4){}

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
//Variable buff_str2 is a pointer to the str to sort here, and
//no length is needed, as there's only one '\n' character
//(no pointer is used, to avoid modifying the original string)
void sorter_fctn(char buff_str2[]){
  //First, strip the spaces in string
  strip_spaces(buff_str2);
  //After stripping, a reordering is necessary
  //Here, I only make use of bubble sort algoritm
  bubble_sort(buff_str2);
  quick_sort(buff_str2);
}

int main(){
  printf("\n");
  printf("Program to test timing for chars organization.\n\n");

  FILE *file_tmp = fopen("./inputfile.txt", "r");

  //Printing first line in file
  if (file_tmp != NULL) {
    //String to store the file's first line
    char buff_str[1000];
    while(fgets(buff_str, sizeof buff_str, file_tmp)!= NULL)
    {
      //Preparing general setting here for reading the whole file
      //...but in case of reading only the first line from file,
      //just the following line applies
      break;
    }
    fclose(file_tmp);

    //This function sorts the original string by both the bubble
    //sort and the quick, and then writes the final results to
    //different output files
    sorter_fctn(buff_str);
    printf("...output written to file ./outfile.txt");
  }
  else {
    //printing error (when reading file) message on stderr
    perror("Something went wrong.");
  }
  printf("\n\n");
}
