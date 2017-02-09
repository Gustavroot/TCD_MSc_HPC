#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;

class RockBand {

public:

  //RockBand( vector<string> members );
  //RockBand( vector<string> members, vector<string> song_list );
  
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
  
  vector<string> set_list;
  vector<string> member_list;
  vector<string> song_list;
  
};





int main(){

  RockBand TheDarkness; //an object: an instance of a class

  TheDarkness.add_member( "Justin" ); //member functions can access private data
  TheDarkness.print_members();
 
  //cout << TheDarkness.member_list[0] << endl; //cannot access private data externally

  RockBand GreenDay; //another object: another instance of the same class

  GreenDay.add_member( "Billie Joe" );
  GreenDay.print_members();

}
