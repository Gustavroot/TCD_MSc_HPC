#include "grid.h"

//constructor
Grid::Grid(int number_points_){

  //TODO: initial checkings

  //TODO: set some Grid values

  number_points = number_points_;

  //TODO: initialize the grid: the Grid consists of a vector of vectors,
  // and each vector corresponding to each point, consists of a set of
  //fishes
  cout << "Initialized grid!" << endl;
}


//destructor
Grid::~Grid(){

  //TODO: carefully write the destruction of the grid here

  cout << "Grid destroyed!" << endl;
}

int Grid::get_nr_points(){
  return number_points;
}

void Grid::randomize(int n_m, int n_t, int n_s){

  //TODO: randomly fill the grid with n_m, n_t, n_s

}
