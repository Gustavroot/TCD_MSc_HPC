#include "user.h"



#define MAX_COL 6
#define MAX_ROW 7


int main(){

  int player = 1;

  string usr_input;
  char usr_input_char;
  int usr_input_col;
  
  int next_turn;
  int move_state;

  cout << "\nInitializing system..." << endl;

  //creating board
  Board board1 = Board(MAX_ROW, MAX_COL);

  //create 2 users
  //user #1 is always non-algorithmic
  cout << "\n**insert name for player 1:" << endl;
  getline(cin, usr_input);
  User user1 = User(usr_input, 1, '*');
  cout << "\n**play against the computer? (y/n)" << endl;
  getline(cin, usr_input);
  if(usr_input == "y"){
    player = -1;
  }
  else if(usr_input == "n"){
    player = 1;
  }
  else{
    cout << "Wrong input! Exiting now." << endl;
    return 0;
  }
  usr_input = "computer";
  //in case of not against computer, ask for name
  if(player == 1){
    cout << "\n**insert second player's name:" << endl;
    getline(cin, usr_input);
    
  }
  User user2 = User(usr_input, player, '+');

  //setting first player!
  next_turn = 1;

  cout << endl << "Play:" << endl << endl;
  board1.display();
  
  char winner_token = ' ';

  while(1){

    if(next_turn == 1){
      //passing the board as a param to 'next_move()'
      //sets the stage for multi-board games
      move_state = user1.next_move(board1);
    }
    else if(next_turn == -1){
      move_state = user2.next_move(board1);
    }
    
    if(move_state == 1){
      cout << "Exit from user input!" << endl;
      return 0;
    }
    
    if(move_state == 2){
      cout << "couldn't allocate new token! play again..." << endl;
    }
    else{
      next_turn *= -1;
    }

    board1.display();
    
    if(board1.check_if_end(&winner_token)){
      cout << "\nEnd of game! Winner: ";
      if(winner_token == '*'){
        cout << user1.name;
      }
      else{
        cout << user2.name;
      }
      cout << endl;
      return 0;
    }
  }

  cout << endl << endl;

  return 0;
}
