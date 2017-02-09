#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <cstdlib>

using namespace std;

class RockBand {

public:

  //constructor
  RockBand( string name );

  //destructor
  ~RockBand();
  
  void play_song(){
    if (song_list.size() != 0){
      int random_number = rand() % song_list.size();
      cout << "Play " << song_list[random_number] << "..." << endl;
    } else {
      cerr << "No songs" << endl; exit(1);
    }
  };
  
  void write_album();
  
  void add_song();

  void add_member( string new_name ){
    member_list.push_back( new_name );
  };

  void print_members(){
    for (int i=0; i<member_list.size(); ++i){
      cout << member_list[i] << endl;
    }
  }

  
private:

  string name;
  vector<string> set_list;
  vector<string> member_list;
  vector<string> song_list;

  complex<double> *ptr;
  
};


RockBand::RockBand( string name_ ){
  name = name_; //copies initialisation value to private variable
  ptr = new complex<double>;
};

RockBand::~RockBand( ){
  delete ptr;
};

int main(){

  while( true ){
    RockBand TheDarkness( "The Darkness" ); 
    TheDarkness.add_member( "Frankie" ); //member functions can access private data
    //TheDarkness.print_members();
  }

}
