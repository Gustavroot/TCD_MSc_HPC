#include<iostream>
#include<vector>
#include<math.h>
#include<map>

using namespace std;

struct general_thing {

  double mydouble;
  string mystring;
};

int main(){

  std::map< int, double > my_map;

  int i=1;
  double x=3.14;

  //insert to map
  for (int j=0; j<10; j++){
    my_map.insert( make_pair( j, x*j ) );
  }

  //manual linear search
  for(  std::map< int, double >::const_iterator it=my_map.begin(); it!=my_map.end(); it++){
    if( (*it).first == 5 ){
      cout <<  (*it).second << endl;
    }
  }

  //how to properly search a map (comparator sorts keys, find uses binary tree)
  std::map< int, double >::iterator iter = my_map.find(5);
  if ( iter!= my_map.end() ){
    cout << (*iter).second << endl;
  }
  
  //more concise:
  cout << (  *(my_map.find(5))  ).second << endl;
  cout << my_map.find(5)->second << endl;

  
  //even more concise
  cout << my_map[5] << endl;
  
  return 0;
}
