#include "fish.h"



//implement external function here to re-assign
//values of indexes due to boundary conditions
void point_boundary_conditions(vector<int> &point){

  //TODO: generalize: don't use the numbers 3, 4 and 5 explicitly!
  for(int i=0; i<3; i++){
    if(point[i] < 0){
      point[i] += 5;
    }
    else if(point[i] > 4){
      point[i] -= 5;
    }
  }
}


//constructor
Fish::Fish(const string & fish_type_, int x_, int y_, int z_){

  //set values of fish
  fish_type = fish_type_;
  steps_without_food = 0;
  
  //cout << "Initialized fish!" << endl;
}

//destructor
Fish::~Fish(){
  //delete[] grid_info;
  //cout << "Fish destroyed!" << endl;
}


void Fish::set_position(int x_, int y_, int z_){
  x = x_;
  y = y_;
  z = z_;
}

vector<int> Fish::get_position(){
  vector<int> vec_buff(3);
  vec_buff[0] = x;
  vec_buff[1] = y;
  vec_buff[2] = z;

  return vec_buff;
}

string Fish::get_type(){
  return fish_type;
}

void Fish::set_type(const string& fish_type_){
  fish_type = fish_type_;
}
