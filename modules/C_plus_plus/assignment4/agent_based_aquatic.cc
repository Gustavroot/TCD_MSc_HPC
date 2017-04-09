#include "external_functions.h"
#include <time.h>



//TODO: enable all 'cout's through an external flag


using namespace std;

int main(){
  cout << endl << "Simulation of evolution of aquatic system." << endl << endl;

  //initializing random seed in general
  srand(time(NULL));

  //creation of the grid.. in this case with 5^3
  Grid grid(125);

  //------------------------------------------------------

  int nr_iterations = 1000;

  //n_m: number of minnows
  int n_m = 1000000, n_t = 200000, n_s = 1000;
  //and other interesting sets:
  //{597878, 14738, 36234} kills the whole tuna!
  //{1048085, 8124, 68519}

  //randomize the grid for those three values
  //grid.randomize(n_m, n_t, n_s);

  //DEBUG print
  //cout << "x-position: " << grid.grid_info[40][0][0].get_position()[0] << endl;

  int general_counter = 0;

  cout << endl << "Each dot represents 100 sweeps." << endl << endl;

  //evolve the system over 1000 sweeps (repeat 5 times as multiple check)
  for(int j=0; j<5; j++){
    //before each simulation, randomize again
    grid.reset();
    grid.randomize(n_m, n_t, n_s);
    //run a complete simulation
    cout << "STATE: " << n_m << ", " << n_t << ", " << n_s << endl;
    for(int i=0; i<nr_iterations; i++){
      evolve_one_sweep(grid, general_counter);
      if(i%100 == 0){cout << "."; cout.flush();}
      //cout << endl << general_counter << "------" << endl;
    }
    cout << "Fishes count: " << grid.fishes_count()[0] << ", ";
    cout << grid.fishes_count()[1] << ", ";
    cout << grid.fishes_count()[2];
    cout << endl;
  }
  cout << endl;

  //------------------------------------------------------
  
  cout << endl << "End of simulation." << endl;

  cout << endl;
  return 0;
}
