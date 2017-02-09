#include "myclass.h"


MyClass::MyClass( string name_ ){
  name = name_;
}

MyClass::~MyClass(){};

void MyClass::print_name(){
  cout << "My class is called " << name << endl;
}

void MyClass::do_a_thing(){
  name = "gone";
}


int main(){

  string my_class_object_name =  " ... something ... ";
  
  MyClass my_object( my_class_object_name );

  my_object.print_name();
  
  my_object.do_a_thing();

  my_object.print_name();

  return 0;
}
