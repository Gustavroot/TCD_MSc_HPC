#include <iostream>
#include <chrono>
#include <random>

using namespace std;

class Board{

  //private:
  
    
  public:

    //Constructor:
    Board(int, int);
    //Destructor
    ~Board();
    
    char* grid_info;
    
    int row_size;
    int col_size;

    void display();
    
    int check_if_end(char*);
};
