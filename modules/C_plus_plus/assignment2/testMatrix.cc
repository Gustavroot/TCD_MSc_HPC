//#include <iostream>
#include "Matrix.h"

int main(){
  std::cout << "\nTesting Matrix class." << std::endl;

  //creation of a sample matrix  
  Matrix matrix1(5,7);
    
  //test of printing methods
  cout << "\nPrint with 'print()':" << endl;
  matrix1.print();
  cout << "\nPrint with 'cout':" << endl;
  cout << matrix1;

  //test of copy constructor
  Matrix matrix2 = matrix1;
  cout << "\nMatrix instanced by copy constructor:" << endl;
  cout << matrix2;
  
  //setting some values in matrix1
  cout << "\nTesting 'set' method:" << endl;
  matrix1.set(4, 1 , 8.4);
  matrix1.set(3, 2 , 4.5);
  matrix1.set(1, 5 , 0.7);
  matrix1.set(2, 3 , 3.456);
  matrix1.set(1, 1 , 67.3);
  cout << "matrix1:" << endl << matrix1;
  cout << "matrix2:" << endl << matrix2;

  //test of 'get' operators
  cout << "\nTest of 'get' operators:" << endl;
  cout << matrix1.elem(3,2) << endl;
  cout << matrix2.elem(3,2) << endl;
  cout << matrix1(3,2) << endl;
  cout << matrix2(3,2) << endl;


  //testing '=' operator
  cout << "\nTest of '=' operator:" << endl;
  matrix2 = matrix1;
  cout << "After matrix2 = matrix1:" << endl;
  cout << "matrix1:" << endl << matrix1;
  cout << "matrix2:" << endl << matrix2;
  //tring to equalize matrices of diff size:
  cout << "\nand trying with matrices of diff size:" << endl;
  Matrix matrix3(2,2);
  matrix3 = matrix2;


  //testing '+' operator
  cout << "\nTest of '+' operator:" << endl;
  matrix2 = matrix1+matrix1;
  cout << "matrix1:" << endl << matrix1;
  cout << "matrix2:" << endl << matrix2;


  //testing 'transpose' method
  cout << "\nTest of 'transpose' method:" << endl;
  cout << "transposed matrix:" << endl << matrix1.transpose();
  
  
  //filling matrix2 with 1s
  cout << "\nSetting matrix2 to only 1s:" << endl;
  for(int i=0; i<5; i++){
    for(int j=0; j<7; j++){
      matrix2.set(i, j, 1);
    }
  }
  cout << matrix2;
  
  
  //testing '*' operator
  //testing 'transpose' method
  cout << "\nTest of '*' operator:" << endl;
  cout << "result of matrix1*matrix2:" << endl << (matrix1.transpose())*matrix2;


  //testing '-' operator
  cout << "\nTest of '-' operator:" << endl;
  matrix2 = matrix2-matrix1;
  cout << "matrix2 (result of matrix2-matrix1):" << endl << matrix2;


  //testing '+='
  cout << "\nTest of '+=' operator:" << endl;
  matrix2+=matrix1;
  cout << "matrix2:" << endl << matrix2;


  //testing '-='
  cout << "\nTest of '-=' operator:" << endl;
  matrix2-=matrix1;
  cout << "matrix2:" << endl << matrix2;

  bool buff_bool;
  //testing comparison operators: '+=' and '-='
  cout << "\nTest of comparison operator '-=' and '+=':" << endl;
  buff_bool = matrix2!=matrix1;
  cout << "matrix2 != matrix1: " << buff_bool << endl;
  buff_bool = matrix2==matrix1;
  cout << "matrix2 == matrix1: " << buff_bool << endl;
  buff_bool = matrix2==matrix2;
  cout << "matrix2 == matrix2: " << buff_bool << endl;

  cout << endl;
  return 0;
}
