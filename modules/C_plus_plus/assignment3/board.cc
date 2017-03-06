#include "board.h"



//EXTRA functions

void aux_check(){}


//CORE Board methods

Board::Board(int x, int y){
  row_size = y;
  col_size = x;
  
  grid_info = new char[row_size*col_size];
  
  for(int i=0; i<row_size*col_size; i++){
    //setting empty spaces at beginning
    grid_info[i] = ' ';
  }
}


//destructor
Board::~Board(){
  delete[] grid_info;
  cout << "Board destroyed!" << endl;
}


void Board::set_value(int, char){}


void Board::display(){
  int i, j;
  cout << endl;
  for(i=0; i<row_size; i++){
    cout << "|" ;
    for(j=0; j<col_size; j++){
      cout << grid_info[i*col_size + j];
      cout << "|" ;
    }
    cout << endl;
  }
  cout << endl;
}


int Board::check_if_end(char* winner_token){

  //checking if end of game
  
  int i, j, k;
  char current = ' ', previous = ' ';
  int counter;

  //check columns
  for(j=0; j<col_size; j++){
    counter = 0;
    for(i=0; i<row_size; i++){
      //*****
      current = grid_info[i*col_size + j];
      if(current != ' ' && current == previous){
        if(counter == 0){counter += 2;}else{counter++;}
      }
      else{counter = 0;}
      //check if 4 tokens are next to each other
      if(counter == 4){*winner_token = current; return 1;}
      //set up for next check
      previous = current;
      //*****
    }
  }
  
  current = ' ', previous = ' ';
  //check rows
  for(i=0; i<row_size; i++){
    counter = 0;
    for(j=0; j<col_size; j++){
      //*****
      current = grid_info[i*col_size + j];
      if(current != ' ' && current == previous){
        if(counter == 0){counter += 2;}else{counter++;}
      }
      else{counter = 0;}
      //check if 4 tokens are next to each other
      if(counter == 4){*winner_token = current; return 1;}
      //set up for next check
      previous = current;
      //*****
    }
  }

  //CHECK DIAGONALS! ----------

  current = ' ', previous = ' ';
  // # right up
  for(j=0; j<(col_size-(4-1)); j++){
    k = j;
    counter = 0;
    for(i=0; i<row_size && k<col_size; i++){
      //*****
      //cout << i << " " << k << ", ";
      current = grid_info[i*col_size + k];
      if(current != ' ' && current == previous){
        if(counter == 0){counter += 2;}else{counter++;}
      }
      else{counter = 0;}
      //check if 4 tokens are next to each other
      if(counter == 4){*winner_token = current; return 1;}
      //set up for next check
      previous = current;
      //*****
      k++;
    }
    //cout << endl;
  }
  
  current = ' ', previous = ' ';
  // # left down
  for(i=1; i<row_size; i++){
    k = i;
    counter = 0;
    for(j=0; j<col_size && k<row_size; j++){
      //*****
      //cout << j << " " << k << ", ";
      current = grid_info[k*col_size + j];
      if(current != ' ' && current == previous){
        if(counter == 0){counter += 2;}else{counter++;}
      }
      else{counter = 0;}
      //check if 4 tokens are next to each other
      if(counter == 4){*winner_token = current; return 1;}
      //set up for next check
      previous = current;
      //*****
      k++;
    }
    //cout << endl;
  }

  current = ' ', previous = ' ';
  // # left up
  for(i=row_size-1; i>=0; i--){
    k = i;
    counter = 0;
    for(j=0; j<col_size && k>=0; j++){
      //*****
      //cout << k << " " << j << ", ";
      current = grid_info[k*col_size + j];
      if(current != ' ' && current == previous){
        if(counter == 0){counter += 2;}else{counter++;}
      }
      else{counter = 0;}
      //check if 4 tokens are next to each other
      if(counter == 4){*winner_token = current; return 1;}
      //set up for next check
      previous = current;
      //*****
      k--;
    }
    //cout << endl;
  }

  current = ' ', previous = ' ';
  // # right down
  for(j=1; j<col_size; j++){
    k = j;
    counter = 0;
    for(i=(row_size-1); i>=0 && k<col_size; i--){
      //*****
      //cout << i << " " << k << ", ";
      current = grid_info[i*col_size + k];
      if(current != ' ' && current == previous){
        if(counter == 0){counter += 2;}else{counter++;}
      }
      else{counter = 0;}
      //check if 4 tokens are next to each other
      if(counter == 4){*winner_token = current; return 1;}
      //set up for next check
      previous = current;
      //*****
      k++;
    }
    //cout << endl;
  }

  return 0;
}
