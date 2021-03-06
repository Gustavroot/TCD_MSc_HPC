#include "external_functions.h"



//TODO: enable all 'cout's through an external flag

//implement following function
//Code imlementing one sweep, with all possible outcomes
//from each sub-step or the sweep
//NOTE: one sweep corresponds to L^3 iterations
void evolve_one_sweep(Grid& grid, int& general_counter){

  //in each of the 1000 moves within one sweep, if
  //multiple moves are possible, then choose one randomly
  //with same probability. Also, if  if any shark or tuna
  //moves 5 times without eating, it dies and must be deleted
  
  int species_index, point, single_fish_index;
  vector<int> grid_point_coordinates(3);

  for(int i=0; i<grid.get_nr_points(); i++){
  
    //cout << endl << general_counter << "------" << endl;
  
    point = rand()%grid.get_nr_points();
    
    //decoding current point into x,y,z values
    coordinates_decode(grid_point_coordinates, point);
  
    //randomly choose a species
    species_index = rand()%3;
    
    //move that species with probability 0.9
    if(((double)rand()/(RAND_MAX)) < 0.9){
      //move the fish!
      //If, in the selected point, there are fishes of the
      //randomly selected species type
      if(grid.grid_info[point][species_index].size() > 0){
        single_fish_index = rand()%grid.grid_info[point][species_index].size();
        grid.move_fish(grid.grid_info[point][species_index][single_fish_index], point, species_index, single_fish_index);
      }
      else{
        //cout << "no fishes to move!" << endl;
      }
    }
    else{
      //cout << "random says no move!" << endl;
    }
    
    //cout << endl;
    //general_counter++;
  }
  
  general_counter++;

  //cout << "Fishes count: " << grid.fishes_count()[0] << ", ";
  //cout << grid.fishes_count()[1] << ", ";
  //cout << grid.fishes_count()[2];
  //cout << endl;
  
  //cout << endl;
}
