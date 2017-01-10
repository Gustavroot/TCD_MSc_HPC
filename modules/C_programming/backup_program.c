#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//TODO: is this solving the general problem of all the possible paths..?


//Convert coordinates from 2D 'real' form to 1D stored info
int convert_dims(int x, int y, int maze_dim){
  return x*maze_dim + y;
}


int recursv_part_maze(char *maze_data, int current_x_val, int current_y_val, int previous_x_val, int previous_y_val, int maze_dim){

  //TODO: reduce all repeated following code with a function

  int next_x_buff, next_y_buff, status_nr;
  //point above
  //..is it possible? (not hitting wall?).. and isn't it the previous point?
  if(current_x_val-1 >= 0 && (current_x_val-1 != previous_x_val && current_y_val != previous_y_val)){
    if(maze_data[convert_dims(current_x_val-1, current_y_val, maze_dim)] == 'G'){
      printf("End of maze reached!\n");
      printf("---> (%d,%d)", current_x_val-1, current_y_val);
      return 1;
    }
    else if(maze_data[convert_dims(current_x_val-1, current_y_val, maze_dim)] == '.'){
      //recursively call this same function, but with above point
      if(recursv_part_maze(maze_data, current_x_val-1, current_y_val, current_x_val, current_y_val) == 1){
        printf("---> (%d,%d)", current_x_val, current_y_val);
        return 1;
      }
      else{return 0;}
    }
  }

  //point at left
  //..is it possible? (not hitting wall?)
  else if(current_y_val-1 >= 0){
  }


  //point below
  //..is it possible? (not hitting wall?)
  else if(current_x_val+1 <= maze_dim){
  }


  //point at right
  //..is it possible? (not hitting wall?)
  else if(current_y_val+1 <= maze_dim){
  }

  else{
    //no solution found
    return 0;
  }

}


//Function for solving the maze
int solve_maze(char *maze_data, int maze_dim){

  int i, sub_counter, S_pos_col, S_pos_row, G_pos_col, G_pos_row;

  //finding the start point
  sub_counter=0;
  for(i=0; i<maze_dim*maze_dim; i++){
    if(maze_data[i] == 'S'){break;}
    if(i%maze_dim == 0){sub_counter++;}
  }
  S_pos_row = sub_counter;
  S_pos_col = i-S_pos_row*maze_dim;

  //finding the goal point
  sub_counter=0;
  for(i=0; i<maze_dim*maze_dim; i++){
    if(maze_data[i] == 'G'){break;}
    if(i%maze_dim == 0){sub_counter++;}
  }
  G_pos_row = sub_counter-1;
  G_pos_col = i-G_pos_row*maze_dim;

  //DEBUG printfs
  //printf("\nStarting point: [%d,%d]", S_pos_row, S_pos_col);
  //printf("\nFinal point: [%d,%d]\n", G_pos_row, G_pos_col);

  //for 'S', scan over the 4 possible surrounding points
  

  //return 1 if the maze was solved correctly
}


//Printing the maze function
void print_maze(char *maze_data, int maze_dim){
  int i;
  for(i=0; i<maze_dim*maze_dim; i++){
    if(i%maze_dim == 0){printf("\n");}
    printf("%c", maze_data[i]);
  }
  printf("\n");
}


int main(){

  int maze_dim, i, counter_maze_data, maze_state;
  char line[25];
  char *maze_data;

  printf("\nProgram for solving maze.\n\n");

  //Reading input file
  FILE *inp_file;
  inp_file = fopen("input_maze.txt", "r");

  //..first line
  fgets(line, sizeof(line), inp_file);
  //..counting dim of maze
  maze_dim = strlen(line)-1;
  if(maze_dim>20){
    printf("Max dim of maze is 20.\n");
    return 0;
  }
  //DEBUG printf:
  //printf("%d\n", maze_dim);
  //..dynamical store of memory for maze info
  maze_data = malloc(maze_dim*maze_dim*sizeof(char));
  for(i=0; i<maze_dim; i++){
    maze_data[i] = line[i];
  }

  //..reading the rest of the file line by line
  counter_maze_data = 1;
  while(fgets(line, sizeof(line), inp_file)) {
    for(i=0; i<maze_dim; i++){
      maze_data[counter_maze_data*maze_dim + i] = line[i];
    }
    //printf("%s", line); 
    counter_maze_data++;
  }
  fclose(inp_file);

  printf("Initial state:\n\n");
  print_maze(maze_data, maze_dim);

  //And solving the maze:
  maze_state = solve_maze(maze_data, maze_dim);
  if(maze_state == 0){printf("\nFINAL STATUS: the maze could not be solved.");}
  else{printf("\nFINAL STATUS: the maze was solved correctly.");}

  printf("\n");
  return 0;
}
