#include "board.h"


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
}


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
  char current, previous;
  int counter;

  cout << "columns!" << endl;
  //check columns
  for(j=0; j<col_size; j++){
    current = ' ', previous = ' ';
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
  
  cout << "rows!" << endl;
  //check rows
  for(i=0; i<row_size; i++){
    current = ' ', previous = ' ';
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

  // # right up
  cout << "right up!" << endl;
  for(j=0; j<col_size; j++){
    k = j;
    current = ' ', previous = ' ';
    counter = 0;
    for(i=0; i<row_size && k<col_size; i++){
      //*****
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
  
  // # left down
  cout << "left down!" << endl;
  for(i=1; i<row_size; i++){
    k = i;
    current = ' ', previous = ' ';
    counter = 0;
    for(j=0; j<col_size && k<row_size; j++){
      //*****
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

  // # left up
  cout << "left up!" << endl;
  for(i=row_size-1; i>=0; i--){
    k = i;
    current = ' ', previous = ' ';
    counter = 0;
    for(j=0; j<col_size && k>=0; j++){
      //*****
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

  // # right down
  cout << "right down!" << endl;
  for(j=1; j<col_size; j++){
    k = j;
    current = ' ', previous = ' ';
    counter = 0;
    for(i=(row_size-1); i>=0 && k<col_size; i--){
      //*****
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
