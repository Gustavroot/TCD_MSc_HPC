#include "user.h"
#include <sstream>
#include <string>

#include <random>

//EXTRA functions

//Using a time-seeded random gaussian with
//mean 4 and deviation 3, to select computer's
//next move
int algorithm(Board& board, char token){

  //TODO: implement defensive method!

  //offensive:
  //algorithm when playing against computer
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(4.0,3.0);

  double number;

  while(1){
    number = distribution(generator);
    if(number>=0 && number<=board.col_size){
      break;
    }
  }
  
  //run through specified colum
  int j = number;
  for(int i=(board.row_size-1); i >= 0; i--){
    if(board.grid_info[i*board.col_size + j] == ' '){
      board.grid_info[i*board.col_size + j] = token;
      return 0;
    }
  }
  
  return 2;
}



//CORE functions: User methods

User::User(string name_i, int type_i, char token_i){
  name = name_i;
  type = type_i;
  token = token_i;
  
  cout << "user " << name << " created!" << endl;
}


//returns 2 if couldn't allocate a new char
//returns 1 to exit from user input
//returns 0 if all good and to continue playing
int User::next_move(Board& board){

  int col;
  string usr_input;
  
  if(type == -1){
    cout << name << "'s turn..." << endl;
    return algorithm(board, token);
  }
  else{
    cout << name << "'s turn: (insert COL number [1,7], ENTER to play same as before, or -1 to exit) ";
    getline(cin, usr_input);
    
    //validatation user input
    if(usr_input.length() != 1){
      return 2;
    }
    else{
      //with the use of this, the function has memory
      istringstream buffer(usr_input);
      buffer >> col;
    }

    //convert to [0,6] to [1,7] representation
    col--;
    
    if(col > (board.col_size-1)){
      return 2;
    }
    
    if(col == -1){
      return 1;
    }

    //run through specified colum
    int j = col;
    for(int i=(board.row_size-1); i >= 0; i--){
      if(board.grid_info[i*board.col_size + j] == ' '){
        board.grid_info[i*board.col_size + j] = token;
        return 0;
      }
    }
  }

  return 2;
}
