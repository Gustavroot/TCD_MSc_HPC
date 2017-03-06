#include <iostream>

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

    //value of column to insert, and char
    void set_value(int, char);

    void display();
    
    int check_if_end(char*);
};
