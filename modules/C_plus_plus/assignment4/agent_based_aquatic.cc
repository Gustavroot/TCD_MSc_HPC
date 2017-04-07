#include "external_functions.h"



using namespace std;

int main(){
  cout << endl << "Simulation of evolution of aquatic system." << endl << endl;

  //creation of the grid
  Grid grid(25);

  int nr_iterations = 1000;

  //cout << grid.get_nr_points() << endl;

  //TODO: instead of the following fixed values, iterate
  //over a set for all three
  int n_m = 1000, n_t = 300, n_s = 75;

  //randomize the grid for those three values
  grid.randomize(n_m, n_t, n_s);

  for(int i=0; i<nr_iterations; i++){
    evolve_one_sweep(grid);
  }


  cout << endl;
  return 0;
}
