#include <iostream>
#include "board.h"


using namespace std;

class User{

  private:
    //1 is normal user, -1 is algorithmic
    int type;
    char token;
    

  public:
  
    string name;
  
    User(string, int, char);
    
    int next_move(Board&);
};
