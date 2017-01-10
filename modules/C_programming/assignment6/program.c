#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
int convert_dims(int x, int y, int maze_dim){}
int aux_function(char *maze_data, int y_buff, int x_buff, int current_x_val, int current_y_val, int maze_dim){}
int recursv_part_maze(char *maze_data, int current_x_val, int current_y_val, int previous_x_val, int previous_y_val, int maze_dim){}
int solve_maze(char *maze_data, int maze_dim){}
void print_maze(char *maze_data, int maze_dim){}
*/


//Convert coordinates from 2D 'real' form to 1D stored info
int convert_dims(int x, int y, int maze_dim){
  return x*maze_dim + y;
}


//Printing the maze function
void print_maze(char *maze_data, int maze_dim){
  int i;
  for(i=0; i<maze_dim*maze_dim; i++){
    if(i%maze_dim == 0){printf("\n");}
    printf("%c", maze_data[i]);
  }
  printf("\n\n");
}



//Auxiliary extra function to optimize amount of code
int aux_function(char *maze_data, int y_buff, int x_buff, int current_x_val, int current_y_val, int maze_dim, char *maze_data_copy){

  if(maze_data[convert_dims(x_buff, y_buff, maze_dim)] == 'G'){
    printf("\nEnd of maze reached! Back-order of steps for solution:\n");
    printf("G=(%d,%d)", x_buff, y_buff);
    return 1;
  }
  else if(maze_data[convert_dims(x_buff, y_buff, maze_dim)] == '.'){
    printf("------possible way!!\n");
    //recursively call this same function, but with above point

    //as it's a possible way, assign a '+' sign
    maze_data_copy[convert_dims(x_buff, y_buff, maze_dim)] = '+';
    print_maze(maze_data_copy, maze_dim);

    if(recursv_part_maze(maze_data, x_buff, y_buff, current_x_val, current_y_val, maze_dim, maze_data_copy) == 1){
      printf("---> (%d,%d)", x_buff, y_buff);
      return 1;
    }
    else{
      //as it failed, assign back a '.' symbol
      maze_data_copy[convert_dims(x_buff, y_buff, maze_dim)] = '.';
      print_maze(maze_data_copy, maze_dim);
      return 0;
    }
  }

}


//This function implements the solution of the maze using recursion
int recursv_part_maze(char *maze_data, int current_x_val, int current_y_val, int previous_x_val, int previous_y_val, int maze_dim, char *maze_data_copy){

  printf("Attempt.. from (%d,%d) to any !\n", current_x_val, current_y_val);
  int x_buff, y_buff, tmp_bool;

  //the maze is solved counterclockwise

  //point above
  x_buff = current_x_val-1;
  y_buff = current_y_val;
  //..is it a possible element? (not hitting at wall?).. and isn't it the previous point?
  printf("---above?\n");
  tmp_bool = !(x_buff == previous_x_val && y_buff == previous_y_val);
  if((x_buff >= 0) && tmp_bool){
    if(aux_function(maze_data, y_buff, x_buff, current_x_val, current_y_val, maze_dim, maze_data_copy) == 1){return 1;}
  }

  //point at left
  x_buff = current_x_val;
  y_buff = current_y_val-1;
  //..is it a possible element? (not hitting at wall?).. and isn't it the previous point?
  printf("---left?\n");
  tmp_bool = !(x_buff == previous_x_val && y_buff == previous_y_val);
  if((y_buff >= 0) && tmp_bool){
    if(aux_function(maze_data, y_buff, x_buff, current_x_val, current_y_val, maze_dim, maze_data_copy) == 1){return 1;}
  }


  //point below
  x_buff = current_x_val+1;
  y_buff = current_y_val;
  //..is it a possible element? (not hitting at wall?).. and isn't it the previous point?
  printf("---below?\n");
  tmp_bool = !(x_buff == previous_x_val && y_buff == previous_y_val);
  if((x_buff < maze_dim) && tmp_bool){
    if(aux_function(maze_data, y_buff, x_buff, current_x_val, current_y_val, maze_dim, maze_data_copy) == 1){return 1;}
  }


  //point at right
  x_buff = current_x_val;
  y_buff = current_y_val+1;
  //..is it a possible element? (not hitting at wall?).. and isn't it the previous point?
  printf("---right?\n");
  tmp_bool = !(x_buff == previous_x_val && y_buff == previous_y_val);
  if((y_buff < maze_dim) && tmp_bool){
    if(aux_function(maze_data, y_buff, x_buff, current_x_val, current_y_val, maze_dim, maze_data_copy) == 1){return 1;}
  }

  printf("**Fail from (%d,%d) to any! Going back...\n", current_x_val, current_y_val);
  //no solution found
  return 0;

}


//Initial shell-function (just the skeleton, because is the function
//recursv_part_maze which implements the solution through the use of
//recursion) for solving the maze
int solve_maze(char *maze_data, int maze_dim, char *maze_data_copy){

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
  return recursv_part_maze(maze_data, S_pos_row, S_pos_col, -5, -5, maze_dim, maze_data_copy);
}



int main(){

  int maze_dim, i, counter_maze_data, maze_state;
  char line[25];
  char *maze_data, *maze_data_copy;

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
  //..and also, making a copy of the maze, for modifying that copy
  maze_data = malloc(maze_dim*maze_dim*sizeof(char));
  maze_data_copy = malloc(maze_dim*maze_dim*sizeof(char));
  for(i=0; i<maze_dim; i++){
    maze_data[i] = line[i];
    maze_data_copy[i] = line[i];
  }

  //..reading the rest of the file line by line
  counter_maze_data = 1;
  while(fgets(line, sizeof(line), inp_file)) {
    for(i=0; i<maze_dim; i++){
      maze_data[counter_maze_data*maze_dim + i] = line[i];
      maze_data_copy[counter_maze_data*maze_dim + i] = line[i];
    }
    //printf("%s", line); 
    counter_maze_data++;
  }
  fclose(inp_file);

  printf("Initial state:\n\n");
  print_maze(maze_data, maze_dim);

  //And solving the maze:
  maze_state = solve_maze(maze_data, maze_dim, maze_data_copy);

  printf("\n");
  if(maze_state == 0){printf("\nFINAL STATUS: the maze could not be solved.\n");}
  else{printf("\nFINAL STATUS: the maze was solved correctly.\n");}

  //Re-printing the maze
  printf("Final state:\n");
  print_maze(maze_data_copy, maze_dim);

  printf("\n");
  return 0;
}
