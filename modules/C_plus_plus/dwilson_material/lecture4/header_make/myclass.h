#include <iostream>
#include <string>

using namespace std;

class MyClass{

 public:
  MyClass( string name_ );
  
  ~MyClass();

  void do_a_thing();

  void print_name();
  
 private:
  string name;

};
