#include "grid.h"


//TODO: enable all 'cout's through an external flag


//Extra functions
void coordinates_decode(vector<int>& vec_buff, int point_nr){
  //0 = x, 1 = y, 2 = z

  vec_buff[2] = point_nr/25;
  vec_buff[0] = (point_nr - vec_buff[2]*25)/5;
  vec_buff[1] = (point_nr - vec_buff[2]*25) - vec_buff[0]*5;
}

int coordinates_encode(const vector<int> vec_buff){
  //TODO: change hardcoded numbers for general grid params
  return vec_buff[0]*5 + vec_buff[1] + vec_buff[2]*25;
}

//constructor
Grid::Grid(int number_points_){

  number_points = number_points_;

  //the Grid consists of a vector with 125 points, each point is
  //a vector with 3 vectors, and each of those 3 is a set of fishes

  //creating the grid info; in each point, vector 0 is for minnows,
  //1 for tuna and 2 for sharks
  vector< vector< vector<Fish> > > grid_buff(number_points);

  //in each point there's a vector with 3 elements
  vector< vector<Fish> > grid_point(3);

  for(int i = 0; i<number_points; i++){
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
  
  //possible moves for tuna
  vector< vector<int> > moves_tuna(12);
  one_move[0] = -1;
  one_move[1] = -1;
  one_move[2] = 0;
  moves_tuna[0] = one_move;
  one_move[0] = -1;
  one_move[1] = 1;
  one_move[2] = 0;
  moves_tuna[1] = one_move;
  one_move[0] = 1;
  one_move[1] = 1;
  one_move[2] = 0;
  moves_tuna[2] = one_move;
  one_move[0] = 1;
  one_move[1] = -1;
  one_move[2] = 0;
  moves_tuna[3] = one_move;
  one_move[0] = -1;
  one_move[1] = 0;
  one_move[2] = -1;
  moves_tuna[4] = one_move;
  one_move[0] = -1;
  one_move[1] = 0;
  one_move[2] = 1;
  moves_tuna[5] = one_move;
  one_move[0] = 1;
  one_move[1] = 0;
  one_move[2] = -1;
  moves_tuna[6] = one_move;
  one_move[0] = 1;
  one_move[1] = 0;
  one_move[2] = 1;
  moves_tuna[7] = one_move;
  one_move[0] = 0;
  one_move[1] = -1;
  one_move[2] = -1;
  moves_tuna[8] = one_move;
  one_move[0] = 0;
  one_move[1] = -1;
  one_move[2] = 1;
  moves_tuna[9] = one_move;
  one_move[0] = 0;
  one_move[1] = 1;
  one_move[2] = -1;
  moves_tuna[10] = one_move;
  one_move[0] = 0;
  one_move[1] = 1;
  one_move[2] = 1;
  moves_tuna[11] = one_move;
  
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
  possible_moves_buff[1] = moves_tuna;
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

    grid_info[rand_buff][0].push_back(minnow);
    //cout << "putting a " << fish_type << " at: " << rand_buff << endl;
    //cout << grid_info[rand_buff][0].size() << endl;
  }

  //tuna
  fish_type = "tuna";
  Fish tuna(fish_type, 0, 0, 0);
  for(int i=0; i<n_t; i++){
    rand_buff = rand()%number_points;

    coordinates_decode(fish_coordinates, rand_buff);
    tuna.set_position(fish_coordinates[0], fish_coordinates[1], fish_coordinates[2]);

    //at each point, index 0 represents tuna
    grid_info[rand_buff][1].push_back(tuna);
    //cout << "putting a " << fish_type << " at: " << rand_buff << endl;
    //cout << grid_info[rand_buff][1].size() << endl;
  }

  //sharks
  fish_type = "shark";
  Fish shark(fish_type, 0, 0, 0);
  for(int i=0; i<n_s; i++){
    rand_buff = rand()%number_points;

    coordinates_decode(fish_coordinates, rand_buff);
    shark.set_position(fish_coordinates[0], fish_coordinates[1], fish_coordinates[2]);

    //at each point, index 0 represents tuna
    grid_info[rand_buff][2].push_back(shark);
    //cout << "putting a " << fish_type << " at: " << rand_buff << endl;
    //cout << grid_info[rand_buff][2].size() << endl;
  }
}



//implement following method
//Depending on the value of 'fish_type', move with equal
//probability into one of the possible next positions
void Grid::move_fish(Fish& fish, int point, int species_index, int single_fish_index){
  //point into which the fish will move
  vector<int> next_point(3), current_position(3), next_point_buff(3);
  Fish fish_buff;
  
  //Variable 'choice' allows to chose between one of all the possible
  //next moves for the fish
  int choice, eaten_tuna = 0, eaten_shark = 0;
  
  //Variable 'scenarios' is what happens when fish arrived at point
  //Indexes within 'scenarios' can be values between 0 and 5
  vector<int> scenarios;

  //For minnows, nearest neighboring sites translates in 6
  //possible points
  if(fish.get_type() == "minnow"){
    //randomly choose one of the 6 possible moves
    choice = rand()%6;

    next_point = possible_moves[0][choice];
  }
  //For tuna, 'planar diagonal sites' means 12 points
  else if(fish.get_type() == "tuna"){
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

  //setting the new position, from the current
  next_point[0] += fish.get_position()[0];
  next_point[1] += fish.get_position()[1];
  next_point[2] += fish.get_position()[2];

  //and fixing 'next_point' due to boundary conditions
  point_boundary_conditions(next_point);

  //cout << "Moving " << fish.get_type() << " from: ";
  //cout << fish.get_position()[0] << ", ";
  //cout << fish.get_position()[1] << ", ";
  //cout << fish.get_position()[2] << " to: ";
  //cout << next_point[0] << ", ";
  //cout << next_point[1] << ", ";
  //cout << next_point[2] << " ??" << endl;

  //move the fish!
  //first: add the fish to the new point
  grid_info[coordinates_encode(next_point)][species_index].push_back(fish);
  
  //second: erase the fish from the previous (current) point
  grid_info[point][species_index].erase(grid_info[point][species_index].begin() + single_fish_index);
  //increment in one the number of steps that the fish has moved
  grid_info[coordinates_encode(next_point)][species_index][grid_info[coordinates_encode(next_point)][species_index].size()-1].total_steps++;
  //and changing its position
  grid_info[coordinates_encode(next_point)][species_index][grid_info[coordinates_encode(next_point)][species_index].size()-1].set_position(next_point[0], next_point[1], next_point[2]);

  //By default, the fish is assumed not to eat, and if the scenario
  //is appropriate, this is reversed
  grid_info[coordinates_encode(next_point)][species_index][grid_info[coordinates_encode(next_point)][species_index].size()-1].steps_without_food++;

  //Before the outcome is chosen: if any shark or tuna moves 5
  //times without eating, it dies and must be deleted
  fish_buff = grid_info[coordinates_encode(next_point)][species_index][grid_info[coordinates_encode(next_point)][species_index].size()-1];
  //cout << "new fish at: " << fish_buff.get_position()[0] << ", ";
  //cout << fish_buff.get_position()[1] << ", ";
  //cout << fish_buff.get_position()[2];
  //cout << endl;
  if(fish_buff.get_type() == "shark" || fish_buff.get_type() == "tuna"){
    if(fish_buff.steps_without_food > 6){
      //cout << "dying " << fish_buff.get_type() << endl;
      grid_info[coordinates_encode(next_point)][species_index].pop_back();
    }
  }

  //evaluate and execute outcomes depending on arrival point

  //more than one scenario is possible.. depending on the state of the arriving point

  //previous to evaluations, get the number of tuna and sharks who have eaten since being born
  for(int i = 0; i<grid_info[coordinates_encode(next_point)][1].size(); i++){
    if(grid_info[coordinates_encode(next_point)][1][i].total_meals > 0){
      eaten_tuna++;
    }
  }
  for(int i = 0; i<grid_info[coordinates_encode(next_point)][2].size(); i++){
    if(grid_info[coordinates_encode(next_point)][2][i].total_meals > 0){
      eaten_shark++;
    }
  }

  //cout << grid_info[coordinates_encode(next_point)][0].size() << endl;
  //cout << grid_info[coordinates_encode(next_point)][1].size() << endl;
  //cout << grid_info[coordinates_encode(next_point)][2].size() << endl;
  
  //at least two minnows
  if(grid_info[coordinates_encode(next_point)][0].size() > 1){
    scenarios.push_back(0);
  }
  //if two of the tuna have eaten
  if(eaten_tuna > 1){
    scenarios.push_back(1);
  }
  //if two of the sharks have eaten
  if(eaten_shark > 1){
    scenarios.push_back(2);
  }
  //if there is one tuna
  if(grid_info[coordinates_encode(next_point)][1].size() == 1){
    scenarios.push_back(3);
  }
  //if there are more than one.. tuna and shark
  if(grid_info[coordinates_encode(next_point)][1].size() > 0 && grid_info[coordinates_encode(next_point)][2].size() > 0){
    scenarios.push_back(4);
  }
  //if there is only one shark
  if(grid_info[coordinates_encode(next_point)][2].size() == 1){
    scenarios.push_back(5);
  }

  //choose one of the possible scenarios
  if(scenarios.size() != 0){
    choice = rand()%scenarios.size();
    choice = scenarios[choice];
    //cout << "scenarios : ";
    for(int i=0; i<scenarios.size(); i++){
      //cout << scenarios[i] << " ";
    }
    //cout << endl;
    //cout << "choice = " << choice << endl;

    //and now implement that scenario case chosen
    if(choice == 0){
      //cout << "**0** produce three additional minnows!!" << endl;
      Fish minnow("minnow", next_point[0], next_point[1], next_point[2]);
      for(int i=0; i<3; i++){
        grid_info[coordinates_encode(next_point)][0].push_back(minnow);
      }
      //cout << "...done!" << endl;
    }
    else if(choice == 1){
      //cout << "**1** produce one additional tuna!!" << endl;
      Fish tuna("tuna", next_point[0], next_point[1], next_point[2]);
      grid_info[coordinates_encode(next_point)][1].push_back(tuna);
      //cout << "...done!" << endl;
    }
    else if(choice == 2){
      //cout << "**2** produce an additional shark!!" << endl;
      Fish shark("shark", next_point[0], next_point[1], next_point[2]);
      grid_info[coordinates_encode(next_point)][2].push_back(shark);
      //cout << "...done!" << endl;
    }
    else if(choice == 3){
      if(grid_info[coordinates_encode(next_point)][0].size() > 0){
        //tuna eats
        grid_info[coordinates_encode(next_point)][1][0].steps_without_food = 0;
        grid_info[coordinates_encode(next_point)][1][0].total_meals++;
      }
      //cout << "**3** delete all minnows at point: " << coordinates_encode(next_point) << endl;
      while(!grid_info[coordinates_encode(next_point)][0].empty()){
        grid_info[coordinates_encode(next_point)][0].pop_back();
      }
      //cout << "...done!" << endl;
    }
    else if(choice == 4){
      //cout << "**4** delete a tuna!" << endl;
      grid_info[coordinates_encode(next_point)][1].pop_back();
      //assume that the the rank 0 shark ate the tuna
      grid_info[coordinates_encode(next_point)][2][0].steps_without_food = 0;
      grid_info[coordinates_encode(next_point)][2][0].total_meals++;
      //cout << "...done!" << endl;
    }
    else if(choice == 5){
      //cout << "**5** delete all minnows at the arrival site and at any neighboring sites" << endl;
      //at next point
      next_point_buff = next_point;
      while(!grid_info[coordinates_encode(next_point_buff)][0].empty()){
        grid_info[coordinates_encode(next_point_buff)][0].pop_back();
      }
      //and at neighbouring points
      //TODO: create external function to reduce all following calls/code
      //x : -1
      next_point_buff[0]++;
      point_boundary_conditions(next_point_buff);
      while(!grid_info[coordinates_encode(next_point_buff)][0].empty()){
        grid_info[coordinates_encode(next_point_buff)][0].pop_back();
      }
      next_point_buff = next_point;
      //x : 1
      next_point_buff[0]--;
      point_boundary_conditions(next_point_buff);
      while(!grid_info[coordinates_encode(next_point_buff)][0].empty()){
        grid_info[coordinates_encode(next_point_buff)][0].pop_back();
      }
      next_point_buff = next_point;
      //y : -1
      next_point_buff[1]++;
      point_boundary_conditions(next_point_buff);
      while(!grid_info[coordinates_encode(next_point_buff)][0].empty()){
        grid_info[coordinates_encode(next_point_buff)][0].pop_back();
      }
      next_point_buff = next_point;
      //y : 1
      next_point_buff[1]--;
      point_boundary_conditions(next_point_buff);
      while(!grid_info[coordinates_encode(next_point_buff)][0].empty()){
        grid_info[coordinates_encode(next_point_buff)][0].pop_back();
      }
      next_point_buff = next_point;
      //z : -1
      next_point_buff[2]++;
      point_boundary_conditions(next_point_buff);
      while(!grid_info[coordinates_encode(next_point_buff)][0].empty()){
        grid_info[coordinates_encode(next_point_buff)][0].pop_back();
      }
      next_point_buff = next_point;
      //z : 1
      next_point_buff[2]--;
      point_boundary_conditions(next_point_buff);
      while(!grid_info[coordinates_encode(next_point_buff)][0].empty()){
        grid_info[coordinates_encode(next_point_buff)][0].pop_back();
      }
      //and the shark ate
      grid_info[coordinates_encode(next_point)][2][0].steps_without_food = 0;
      grid_info[coordinates_encode(next_point)][2][0].total_meals++;
      //cout << "...done!" << endl;
    }
  }
  else{
    //cout << "no scenario.. nothing to do after moving the fish!" << endl;
  }
}

vector<int> Grid::fishes_count(){

  vector<int> total_count(3);
  total_count[0] = 0;
  total_count[1] = 0;
  total_count[2] = 0;

  //minnows
  for(int i=0; i<grid_info.size(); i++){
    total_count[0] += grid_info[i][0].size();
  }

  //tuna
  for(int i=0; i<grid_info.size(); i++){
    total_count[1] += grid_info[i][1].size();
  }

  //sharks
  for(int i=0; i<grid_info.size(); i++){
    total_count[2] += grid_info[i][2].size();
  }

  return total_count;
}

void Grid::reset(){
  //minnows
  for(int i=0; i<grid_info.size(); i++){
    while(!grid_info[i][0].empty()){
      grid_info[i][0].pop_back();
    }
  }

  //tuna
  for(int i=0; i<grid_info.size(); i++){
    while(!grid_info[i][1].empty()){
      grid_info[i][1].pop_back();
    }
  }

  //sharks
  for(int i=0; i<grid_info.size(); i++){
    while(!grid_info[i][2].empty()){
      grid_info[i][2].pop_back();
    }
  }
}
