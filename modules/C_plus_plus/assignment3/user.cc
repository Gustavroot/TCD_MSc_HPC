#include "user.h"
#include <sstream>
#include <string>

//EXTRA functions

void algorithm(Board&){

  //TODO: implement algorithm when playing against computer!

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
    algorithm(board);
    return 0;
  }
  else{
    cout << name << "'s turn: (insert COL number, ENTER to play same as before, or -1 to exit) ";
    getline(cin, usr_input);
    
    //TODO: validate user input!

    //with the use of this, the function has memory
    istringstream buffer(usr_input);
    buffer >> col;
    
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
