#include <iostream>
#include <vector>
#include <stdlib.h>

using namespace std;

//Extra functions
void point_boundary_conditions(vector<int> &);


//Base function inherited by 'Lattice' class
class Fish{

  public:

    Fish();

    //Constructor: receives string with type of fish, and
    //position of the fish
    Fish(const string &, int, int, int);

    //Destructor
    ~Fish();

    //Set fish at specific point of the grid
    void set_position(int, int, int);

    //Physical dimension
    int steps_without_food;

    int total_steps;
    int total_meals;

    vector<int> get_position();
    string get_type();
    void set_type(const string &);

    void destroy();

  private:

    string fish_type;

    int x;
    int y;
    int z;
};
