#include <iostream>


using namespace std;

//Base function inherited by 'Lattice' class
class Fish{

  public:

    //Constructor: receives string with type of fish, and
    //position of the fish
    Fish(string, int, int, int);

    //Destructor
    ~Fish();

    //Set value at specific point of the grid
    void set_position(int, int, int, double);

    //Physical dimension
    int steps_without_food;

    //Implement one move, with equal probability in
    //each possible direction
    void move();

    //Side_length^2 is the number of points in the grid
    //int side_length;

  private:

    string fish_type;

    int x;
    int y;
    int z;
};
