#include <iostream>
#include <iomanip>
//#include <fstream>


using namespace std;

class Matrix{
  private:
    int n_rows;
    int n_cols;
    double* info;



  public:
  
    //core methods
    Matrix(int, int);
    ~Matrix(); 
    Matrix(const Matrix&);
    Matrix& operator=(const Matrix&);
    
    //more specific methods
    void print();
    friend ostream& operator<<(ostream& os, const Matrix&);
    
    void set(int i_row, int j_col, double val);
    
    double elem(int i_col, int j_row);
    double operator()(int i_col, int j_row);

    Matrix& operator+(const Matrix& matr_in);
    Matrix& operator-(const Matrix& matr_in);
    Matrix& operator+=(const Matrix& matr_in);
    Matrix& operator-=(const Matrix& matr_in);
    Matrix& operator!=(const Matrix& matr_in);
    Matrix& operator==(const Matrix& matr_in);
    
    //transpose
    Matrix& transpose();
    
    //multiply
    Matrix& operator*(const Matrix& matr_in);
};
