#include "grid.h"




//Extra functions
void coordinates_decode(vector<int>& vec_buff, int point_nr){
  //0 = x, 1 = y, 2 = z

  vec_buff[2] = point_nr/25;
  vec_buff[0] = (point_nr - vec_buff[2]*25)/5;
  vec_buff[1] = (point_nr - vec_buff[2]*25) - vec_buff[0]*5;
}


//constructor
Grid::Grid(int number_points_){

  number_points = number_points_;

  //the Grid consists of a vector with 125 points, each point is
  //a vector with 3 vectors, and each of those 3 is a set of fishes

  //creating the grid info; in each point, vector 0 is for minnows,
  //1 for tunna and 2 for sharks
  vector< vector< vector<Fish> > > grid_buff(number_points);

  //in each point there's a vector with 3 elements
  vector< vector<Fish> > grid_point(3);

  for(int i; i<number_points; i++){
    grid_buff[i] = grid_point;
  }
  grid_info = grid_buff;
  
  //----------------------------------------------
  //SETTING POSSIBLE FISH MOVES
  
  vector< vector< vector<int> > > possible_moves_buff(3);
  
  vector<int> one_move(3);
  
  //TODO: re-implement moves to optimize amount of code; this
  //implementation obeys simpleness of use. Use a hash table
  //(which enables the possibility of a tree)
  
  //possible moves for minnows
  vector< vector<int> > moves_minnows(6);
  one_move[0] = 1;
  one_move[1] = 0;
  one_move[2] = 0;
  moves_minnows[0] = one_move;
  one_move[0] = -1;
  one_move[1] = 0;
  one_move[2] = 0;
  moves_minnows[1] = one_move;
  one_move[0] = 0;
  one_move[1] = 1;
  one_move[2] = 0;
  moves_minnows[2] = one_move;
  one_move[0] = 0;
  one_move[1] = -1;
  one_move[2] = 0;
  moves_minnows[3] = one_move;
  one_move[0] = 0;
  one_move[1] = 0;
  one_move[2] = 1;
  moves_minnows[4] = one_move;
  one_move[0] = 0;
  one_move[1] = 0;
  one_move[2] = -1;
  moves_minnows[5] = one_move;
  
  //possible moves for tunna
  vector< vector<int> > moves_tunna(12);
  one_move[0] = -1;
  one_move[1] = -1;
  one_move[2] = 0;
  moves_tunna[0] = one_move;
  one_move[0] = -1;
  one_move[1] = 1;
  one_move[2] = 0;
  moves_tunna[1] = one_move;
  one_move[0] = 1;
  one_move[1] = 1;
  one_move[2] = 0;
  moves_tunna[2] = one_move;
  one_move[0] = 1;
  one_move[1] = -1;
  one_move[2] = 0;
  moves_tunna[3] = one_move;
  one_move[0] = -1;
  one_move[1] = 0;
  one_move[2] = -1;
  moves_tunna[4] = one_move;
  one_move[0] = -1;
  one_move[1] = 0;
  one_move[2] = 1;
  moves_tunna[5] = one_move;
  one_move[0] = 1;
  one_move[1] = 0;
  one_move[2] = -1;
  moves_tunna[6] = one_move;
  one_move[0] = 1;
  one_move[1] = 0;
  one_move[2] = 1;
  moves_tunna[7] = one_move;
  one_move[0] = 0;
  one_move[1] = -1;
  one_move[2] = -1;
  moves_tunna[8] = one_move;
  one_move[0] = 0;
  one_move[1] = -1;
  one_move[2] = 1;
  moves_tunna[9] = one_move;
  one_move[0] = 0;
  one_move[1] = 1;
  one_move[2] = -1;
  moves_tunna[10] = one_move;
  one_move[0] = 0;
  one_move[1] = 1;
  one_move[2] = 1;
  moves_tunna[11] = one_move;
  
  //possible moves for sharks
  vector< vector<int> > moves_sharks(24);
  one_move[0] = -2;
  one_move[1] = 0;
  one_move[2] = 1;
  moves_sharks[0] = one_move;
  one_move[0] = -2;
  one_move[1] = 0;
  one_move[2] = -1;
  moves_sharks[1] = one_move;
  one_move[0] = -2;
  one_move[1] = 1;
  one_move[2] = 0;
  moves_sharks[2] = one_move;
  one_move[0] = -2;
  one_move[1] = -1;
  one_move[2] = 0;
  moves_sharks[3] = one_move;
  one_move[0] = 2;
  one_move[1] = 0;
  one_move[2] = 1;
  moves_sharks[4] = one_move;
  one_move[0] = 2;
  one_move[1] = 0;
  one_move[2] = -1;
  moves_sharks[5] = one_move;
  one_move[0] = 2;
  one_move[1] = 1;
  one_move[2] = 0;
  moves_sharks[6] = one_move;
  one_move[0] = 2;
  one_move[1] = -1;
  one_move[2] = 0;
  moves_sharks[7] = one_move;
  one_move[0] = 0;
  one_move[1] = -2;
  one_move[2] = 1;
  moves_sharks[8] = one_move;
  one_move[0] = 0;
  one_move[1] = -2;
  one_move[2] = -1;
  moves_sharks[9] = one_move;
  one_move[0] = 1;
  one_move[1] = -2;
  one_move[2] = 0;
  moves_sharks[10] = one_move;
  one_move[0] = -1;
  one_move[1] = -2;
  one_move[2] = 0;
  moves_sharks[11] = one_move;
  one_move[0] = 0;
  one_move[1] = 2;
  one_move[2] = 1;
  moves_sharks[12] = one_move;
  one_move[0] = 0;
  one_move[1] = 2;
  one_move[2] = -1;
  moves_sharks[13] = one_move;
  one_move[0] = 1;
  one_move[1] = 2;
  one_move[2] = 0;
  moves_sharks[14] = one_move;
  one_move[0] = -1;
  one_move[1] = 2;
  one_move[2] = 0;
  moves_sharks[15] = one_move;
  one_move[0] = 0;
  one_move[1] = 1;
  one_move[2] = -2;
  moves_sharks[16] = one_move;
  one_move[0] = 0;
  one_move[1] = -1;
  one_move[2] = -2;
  moves_sharks[17] = one_move;
  one_move[0] = 1;
  one_move[1] = 0;
  one_move[2] = -2;
  moves_sharks[18] = one_move;
  one_move[0] = -1;
  one_move[1] = 0;
  one_move[2] = -2;
  moves_sharks[19] = one_move;
  one_move[0] = 0;
  one_move[1] = 1;
  one_move[2] = 2;
  moves_sharks[20] = one_move;
  one_move[0] = 0;
  one_move[1] = -1;
  one_move[2] = 2;
  moves_sharks[21] = one_move;
  one_move[0] = 1;
  one_move[1] = 0;
  one_move[2] = 2;
  moves_sharks[22] = one_move;
  one_move[0] = -1;
  one_move[1] = 0;
  one_move[2] = 2;
  moves_sharks[23] = one_move;

  possible_moves_buff[0] = moves_minnows;
  possible_moves_buff[1] = moves_tunna;
  possible_moves_buff[2] = moves_sharks;
  possible_moves = possible_moves_buff;
  
  //----------------------------------------------
  
  cout << "Initialized grid!" << endl;
}


//destructor
Grid::~Grid(){
  cout << "Grid destroyed!" << endl;
}

int Grid::get_nr_points(){
  return number_points;
}

void Grid::randomize(int n_m, int n_t, int n_s){

  //randomly fill the grid with n_m, n_t, n_s

  int rand_buff;

  string fish_type;
  vector<int> fish_coordinates(3);
  
  //TODO: reduce then next three for loops to just one

  //minnows
  fish_type = "minnow";
  Fish minnow(fish_type, 0, 0, 0);
  for(int i=0; i<n_m; i++){
    rand_buff = rand()%number_points;

    coordinates_decode(fish_coordinates, rand_buff);
    minnow.set_position(fish_coordinates[0], fish_coordinates[1], fish_coordinates[2]);

    //at each point, index 0 represents minnows
    grid_info[rand_buff][0].push_back(minnow);
  }

  //tunna
  fish_type = "tunna";
  Fish tunna(fish_type, 0, 0, 0);
  for(int i=0; i<n_t; i++){
    rand_buff = rand()%number_points;

    coordinates_decode(fish_coordinates, rand_buff);
    tunna.set_position(fish_coordinates[0], fish_coordinates[1], fish_coordinates[2]);

    //at each point, index 0 represents tunna
    grid_info[rand_buff][1].push_back(tunna);
  }

  //sharks
  fish_type = "shark";
  Fish shark(fish_type, 0, 0, 0);
  for(int i=0; i<n_s; i++){
    rand_buff = rand()%number_points;

    coordinates_decode(fish_coordinates, rand_buff);
    shark.set_position(fish_coordinates[0], fish_coordinates[1], fish_coordinates[2]);

    //at each point, index 0 represents tunna
    grid_info[rand_buff][2].push_back(shark);
  }
}






//TODO: implement following method
//Depending on the value of 'fish_type', move with equal
//probability into one of the possible next positions
void Grid::move_fish(Fish& fish){
  vector<int> next_point(3);
  int choice;

  //For minnows, nearest neighboring sites translates in 6
  //possible points
  if(fish.get_type() == "minnow"){
    //randomly choose one of the 6 possible moves
    choice = rand()%6;

    next_point = possible_moves[0][choice];
  }
  //For tunna, 'planar diagonal sites' means 12 points
  else if(fish.get_type() == "tunna"){
    //randomly choose one of the 6 possible moves
    choice = rand()%12;

    next_point = possible_moves[1][choice];
  }
  //In the case of sharks, L moves means 24 possible points
  else if(fish.get_type() == "shark"){
    //randomly choose one of the 6 possible moves
    choice = rand()%24;

    next_point = possible_moves[2][choice];
  }

  next_point[0] += fish.get_position()[0];
  next_point[1] += fish.get_position()[1];
  next_point[2] += fish.get_position()[2];

  //and fixing 'next point' due to boundary conditions
  point_boundary_conditions(next_point);
  
  //TODO: evaluate and execute outcomes depending on arrival point

  cout << "Moving " << fish.get_type() << " from: ";
  cout << fish.get_position()[0] << ", ";
  cout << fish.get_position()[1] << ", ";
  cout << fish.get_position()[2] << " to: ";
  cout << next_point[0] << ", ";
  cout << next_point[1] << ", ";
  cout << next_point[2] << " ??" << endl;
}
