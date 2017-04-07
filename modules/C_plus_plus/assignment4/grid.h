#include "fish.h"
#include <vector>


//Extra functions
void coordinates_decode(vector<int> &, int);


using namespace std;

//Base function inherited by 'Lattice' class
class Grid{

  public:

    //Constructor: receives value of side_length and dim
    Grid(int);

    //Destructor
    ~Grid();

    //Set value at specific point of the grid
    void insert_fish(int, int, int);

    //print the grid
    void print();

    //Return total sum: just performs a summation
    //of the values in each point of the grid
    //double get_total_sum();

    //print a small report on total info of one point
    void get_point_info(int, int, int);

    //'get' operator
    //double operator()(int, int, int);

    int get_nr_points();

    //vector storing the grid info
    vector< vector< vector<Fish> > > grid_info;

    vector< vector< vector<int> > > possible_moves;

    //randomizing the grid for a set of three possible values
    //for amounts of fishes
    void randomize(int, int, int);

    //Implement one fish move, with equal probability in
    //each possible direction
    void move_fish(Fish&);

  protected:

    //Grid_info points to 1D array, independent of physical dim
    //double *grid_info;

    int number_points;

    //TODO: the Grid consists of a vector of vectors, and each
    //vector corresponding to each point, consists of a set of
    //fishes
};
