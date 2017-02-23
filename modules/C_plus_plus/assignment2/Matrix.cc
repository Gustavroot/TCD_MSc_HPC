#include "Matrix.h"


//constructor
Matrix::Matrix(int nrows, int ncols){
  n_rows = nrows;
  n_cols = ncols;
  info = new double[n_rows*n_cols];
}


//destructor
Matrix::~Matrix(){
  delete[] info;
}


//print method
void Matrix::print(){
  int i;
  for(i = 0; i<n_rows*n_cols; i++){
    cout << setprecision(2) << info[i] << "\t";
    if((i+1)%n_cols == 0 && i!=0){cout << endl;}
  }
}
//IMPORTANT: this is not a member function, is an independent
//function to the class Matrix, but a friend, that's why
//not 'Matrix::' prefix
ostream& operator<<(ostream& os, const Matrix& matrix_in){
  int i;
  for(i = 0; i<matrix_in.n_rows*matrix_in.n_cols; i++){
    os << setprecision(2) << matrix_in.info[i] << "\t";
    if((i+1)%matrix_in.n_cols == 0 && i!=0){os << endl;}
  }
  
  return os;
}


//copy constructor
Matrix::Matrix(const Matrix& matr_in){
  n_rows = matr_in.n_rows;
  n_cols = matr_in.n_cols;
  {
    if(matr_in.info != 0){
      info = new double[n_rows*n_cols];
      for(int i=0; i<n_rows*n_cols; i++){
        info[i] = matr_in.info[i];
      }
    }
    else{
      info = 0;
    }
  }
}


//'set' operator
void Matrix::set(int i_row, int j_col, double val){
  info[i_row*n_cols + j_col] = val;
}


//'get' operators
double Matrix::elem(int i_row, int j_col){
  return info[i_row*n_cols + j_col];
}
double Matrix::operator()(int i_row, int j_col){
  return info[i_row*n_cols + j_col];
}


//assignment operator
Matrix& Matrix::operator=(const Matrix& matr_in){
  //if matrices sizes don't match, then can't be equaled, and
  //then return 'this'
  if(n_rows != matr_in.n_rows || n_cols != matr_in.n_cols){
    cout << "Matrices dims don't match!" << endl;
    return *this;
  }
  if(this != &matr_in){
    n_rows = matr_in.n_rows;
    n_cols = matr_in.n_cols;
    for(int i=0; i<n_rows*n_cols; i++){
      info[i] = matr_in.info[i];
    }
  }
  return *this;
}


//transpose
Matrix& Matrix::transpose(){

  Matrix matr_buff = *this;
  
  //swap of info in matrices
  int tmp = n_rows;
  n_rows = n_cols;
  n_cols = tmp;  
  
  for (int i = 0; i<n_rows; i++){
    for (int j = 0; j<n_cols; j++){
      (*this).set(i,j, matr_buff(j,i));
    }
  }
  return *this;
}


//multiply
Matrix& Matrix::operator*(const Matrix& matr_in){
  //if matrices sizes don't match, then can't be multiplied, and
  //then return 'this'
  if(n_cols != matr_in.n_rows){
    cout << "Matrices dims don't match!" << endl;
    return *this;
  }
  Matrix matr_buff = *this;
  
  double buff_sum;
  for(int i=0; i<matr_buff.n_rows; i++){
    for(int j=0; j<matr_in.n_cols; j++){
      buff_sum = 0;
      for(int k=0; k<matr_buff.n_cols; k++){
        buff_sum += matr_buff.info[i*matr_buff.n_cols+k] * matr_in.info[k*matr_in.n_cols + j];
      }
      (*this).set(i, j, buff_sum);
    }
  }
  return *this;
}


//addition
Matrix& Matrix::operator+(const Matrix& matr_in){
  //if matrices sizes don't match, then can't be added, and
  //then return 'this'
  if(n_rows != matr_in.n_rows || n_cols != matr_in.n_cols){
    cout << "Matrices dims don't match!" << endl;
    return *this;
  }
  //else, add and return the result
  Matrix matr_buff = *this;
  
  for(int i=0; i<matr_buff.n_rows*matr_buff.n_cols; i++){
    (*this).info[i] = matr_in.info[i] + matr_buff.info[i];
  }
  return *this;
}


//subtraction
Matrix& Matrix::operator-(const Matrix& matr_in){
  //if matrices sizes don't match, then can't be subtracted, and
  //then return 'this'
  if(n_rows != matr_in.n_rows || n_cols != matr_in.n_cols){
    cout << "Matrices dims don't match!" << endl;
    return *this;
  }
  //else, add and return the result
  Matrix matr_buff = *this;

  for(int i=0; i<matr_buff.n_rows*matr_buff.n_cols; i++){
    (*this).info[i] = matr_buff.info[i] - matr_in.info[i];
  }
  return *this;
}


//second form for addition
Matrix& Matrix::operator+=(const Matrix& matr_in){
  *this = *this + matr_in;
  return *this;
}


//second form for subtraction
Matrix& Matrix::operator-=(const Matrix& matr_in){
  *this = *this - matr_in;
  return *this;
}


//bool comparison operators
bool Matrix::operator==(const Matrix& matr_in){
  //if matrices sizes don't match, then can't be subtracted, and
  //then return 'this'
  if(n_rows != matr_in.n_rows || n_cols != matr_in.n_cols){
    cout << "Matrices dims don't match!" << endl;
    return 0;
  }
  for(int i=0; i<n_rows*n_cols; i++){
    if((*this).info[i] != matr_in.info[i]){
      return 0;
    }
  }
  return 1;
}

bool Matrix::operator!=(const Matrix& matr_in){
  return ! (*this == matr_in);
}
