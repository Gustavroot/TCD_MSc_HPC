#include "external_functions.h"
#include <time.h>



using namespace std;

int main(){
  cout << endl << "Simulation of evolution of aquatic system." << endl << endl;

  //initializing random seed in general
  srand(time(NULL));

  //creation of the grid.. in this case with 5^3
  Grid grid(125);

  //------------------------------------------------------

  int nr_iterations = 1; //TODO: change for 1000

  //TODO: instead of the following fixed values, iterate
  //over a set for all three
  int n_m = 1000, n_t = 300, n_s = 75;

  //randomize the grid for those three values
  grid.randomize(n_m, n_t, n_s);

  //DEBUG print
  //cout << "x-position: " << grid.grid_info[40][0][0].get_position()[0] << endl;

  //evolve the system over 1000 sweeps
  for(int i=0; i<nr_iterations; i++){
    evolve_one_sweep(grid);
  }

  //------------------------------------------------------

  cout << endl << "End of simulation." << endl;

  cout << endl;
  return 0;
}
