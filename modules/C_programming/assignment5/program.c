#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//struct for a doubly-linked list
/*The longest word in any of the major English language
dictionaries is pneumonoultramicroscopicsilicovolcanoconiosis */
struct node {
  char word[45];
  struct node *next;
  struct node *previous;
};
typedef struct node NODE;


//function which, given the double-linked list
//and a word, deletes all appearances of that word from the list
void delete_word(NODE *doub_l_list, char *word_to_del){
  NODE *buff;
  buff = doub_l_list;

  while(buff->next != 0){
    if(strcmp(buff->word,word_to_del) == 0){
      //in case root contains the word to delete
      if(buff->previous == NULL){
        buff = buff->next;
        free(buff->previous);
        buff->previous = NULL;
      }
      else{
        buff = buff->next;
        ((buff->previous)->previous)->next = buff;
        free(buff->previous);
        buff->previous = NULL;
      }
    }
    else{buff = buff->next;}
  }
}


//Function for debugging: printing a list
void print_doub_list(NODE *doub_l_list){
  NODE *buff;
  buff = doub_l_list;

  while(buff->next != 0){
    printf("%s", buff->word);
    buff = buff->next;
  }
}


//function to load the list from an external file
void load_list(NODE *root, char *file_name){
  FILE *file;
  char line[50];

  int list_counter = 0;

  file = fopen(file_name, "r");
  if (file == NULL)
    exit(EXIT_FAILURE);

  NODE *buff1, *buff2;

  while (fgets(line, sizeof(line), file)) {
    if(list_counter == 0){
      buff1 = (NODE *) malloc(sizeof(NODE));
      //storing the word
      strcpy(root->word, line);
      //linking elements in list
      root->next = buff1;
      buff1->previous = root;
    }
    else{
      //In case we're not treating the root, then the steps are:
      //create buff2, finish filling buff1, link, assign buff2 to buff1
      buff2 = (NODE *) malloc(sizeof(NODE));
      buff1->next = buff2;
      buff2->previous = buff1;
      strcpy(buff1->word, line);
      buff1 = buff2;
    }
    //DEBUG print
    //printf("%s", line);
    list_counter++;
  }
  fclose(file);

  //Finally, we can take the last element and assign data to it,
  //for which we assign the total number of elements in the list
  if(list_counter == 1){
    char buff_str[5];
    sprintf(buff_str, "%d", list_counter);
    strcpy(buff1->word, buff_str);
    buff1->previous = 0;
  }
  //otherwise
  else{
    char buff_str[5];
    sprintf(buff_str, "%d", list_counter);
    strcpy(buff2->word, buff_str);
    //last element in list points to NULL
    buff2 = NULL;
    buff1->next = buff2;
    //buff1->previous = buff2;
  }
}


int main(){
  //First element of list:
  NODE *root;
  root = (NODE *) malloc(sizeof(NODE));
  root->previous = NULL;

  //fill list, calling function 'load_list'
  load_list(root, "input_list.txt");

  //before deleting elements
  printf("\n****Before deleting words:\n\n");
  print_doub_list(root);

  printf("\n****Words to delete:\n\n");

  FILE *file;
  char line[50];
  file = fopen("remove_word.txt", "r");
  if (file == NULL)
    exit(EXIT_FAILURE);
  //for each line in file remove_word.txt, delete that appearance
  while (fgets(line, sizeof(line), file)) {
    printf("%s", line);
    delete_word(root, line);
  }
  printf("\n");

  printf("****After deleting the words:\n\n");
  print_doub_list(root);

  fclose(file);

  //store the final list to an output file
  FILE *output_file;
  output_file = fopen("output_list.txt", "w");
  if (file == NULL)
    exit(EXIT_FAILURE);

  //iterating over list
  NODE *buff;
  buff = root;
  while(buff->next != 0){
    fprintf(output_file, "%s", buff->word);
    buff = buff->next;
  }

  fclose(output_file);

  return 0;
}
