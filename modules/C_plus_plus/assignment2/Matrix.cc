#include "Matrix.h"


//TODO: check (send email to David Wilson asking these)
//	-- are the uses of 'static' the appropriate here?
//	-- should I implement '=' with move? what happens in
//	cases where I just do A = B?


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
  static Matrix matr_buff(n_cols, n_rows);
  for(int i=0; i<n_rows; i++){
    for(int j=0; j<n_cols; j++){
      matr_buff.set(j, i, (*this).elem(i,j));
    }
  }
  return matr_buff;
}

//multiply
Matrix& Matrix::operator*(const Matrix& matr_in){
  //if matrices sizes don't match, then can't be multiplied, and
  //then return 'this'
  if(n_cols != matr_in.n_rows){
    cout << "Matrices dims don't match!" << endl;
    return *this;
  }
  static Matrix matr_buff(n_rows, matr_in.n_cols);
  double buff_sum;
  for(int i=0; i<n_rows; i++){
    for(int j=0; j<matr_in.n_cols; j++){
      buff_sum = 0;
      for(int k=0; k<n_cols; k++){
        buff_sum += info[i*n_cols+k] * matr_in.info[k*matr_in.n_cols + j];
      }
      matr_buff.set(i, j, buff_sum);
    }
  }
  return matr_buff;
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
  static Matrix matr_buff(n_rows, n_cols);
  for(int i=0; i<n_rows*n_cols; i++){
    matr_buff.info[i] = matr_in.info[i] + info[i];
  }
  return matr_buff;
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
  static Matrix matr_buff(n_rows, n_cols);
  for(int i=0; i<n_rows*n_cols; i++){
    matr_buff.info[i] = info[i] - matr_in.info[i];
  }
  return matr_buff;
}




//TODO: implement and test (@ testMatrix.cc) following methods:
Matrix& Matrix::operator+=(const Matrix& matr_in){}
Matrix& Matrix::operator-=(const Matrix& matr_in){}
Matrix& Matrix::operator!=(const Matrix& matr_in){}
Matrix& Matrix::operator==(const Matrix& matr_in){}
