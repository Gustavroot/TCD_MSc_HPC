#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/*
NOTE 0: how to compile:
gcc program.c -lm -o program
-- the -lm flag is because, in case of
using array calls such as n[i] inside the
math function pow(xx,xx), the linking process
(to the math library) is not done correctly --
*/

/*
NOTE 1: a check for positive or negative numbers
wasn't included explicitely here, but the program
takes into account that, as it doesn't allow any
non-numeric characters in the inserted strings, therefore
an inclusion of that explicit check would be redundant
*/

/*
NOTE 2: a small modification was implemented, in the
declaration of functions to check if numbers are Armstrong,
as receiving int and then converting back to str would
result in extra complexity of the program
*/

//Definition of function to check if number is 3-Arms like
int checkThreeDigitArmstrongNumber(char n[]){
  int sum_cubes = 0;
  //Fixed number 3 can be used here, due to previous branching
  for(int i=0; i<3; i++){
    //'0' has the ascii threshold value to create the rest of ints
    sum_cubes += pow(n[i]-'0', 3);
  }
  int inserted_num = atoi(n);
  //Final comparison to check if Armstrong
  if(inserted_num == sum_cubes){return 1;}
  else{return 0;}
}

//Definition of function to check if number is k-Arms like
int checkArmstrongNumber(char n[]){
  int sum_powers = 0;
  //Variable number of characters in string
  int length_str = strlen(n);
  for(int i=0; i<length_str; i++){
    //'0' has the ascii threshold value to create the rest of ints
    sum_powers += pow(n[i]-'0', length_str);
  }
  int inserted_num = atoi(n);
  //Final comparison to check if Armstrong
  if(inserted_num == sum_powers){return 1;}
  else{return 0;}
}

//Function to check if string is an int
int checkIfInt(char s[]){
  int length = strlen(s);
  for (int i=0; i<length; i++)
    if (!isdigit(s[i]))
    {
      return 0;
    }
  return 1;
}

int main(){
  int max_str_length = 100;

  //Variables for inserted values from the user
  char three_digit_str[max_str_length];
  char k_digit_str[max_str_length];

  //Requesting data to test 3-digit function
  printf("\nProgram for checking if a number is an Armstrong number.\n\n");
  printf("1-- Insert the 3-digit number to check: ");
  scanf("%s", three_digit_str);

  //This branching checks if int or not
  if(checkIfInt(three_digit_str) == 0){
    printf("...inserted data is not int.\n");
  }
  else if(strlen(three_digit_str) != 3){
    printf("...inserted number is not 3-digit long.\n");
  }
  else{
    //Check if Armstrong
    if(checkThreeDigitArmstrongNumber(three_digit_str) == 1){
      printf("...inserted number is Armstrong!\n");
    }
    else{
      printf("...inserted number is not Armstrong!\n");
    }
  }

  //Requesting data to test k-digit function
  printf("\n");
  printf("2-- Insert the k-digit (k>0) number to check: ");
  scanf("%s", k_digit_str);

  //This branching checks if int or not
  if(checkIfInt(k_digit_str) == 0){
    printf("...inserted data is not int.\n");
  }
  else{
    //Check if Armstrong
    if(checkArmstrongNumber(k_digit_str) == 1){
      printf("...inserted number is Armstrong!\n");
    }
    else{
      printf("...inserted number is not Armstrong!\n");
    }
  }

  printf("\n");
}
