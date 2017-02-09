#include <iostream>

using namespace std;

//Compilation instructions
//	$ g++ hw1_gustavo.cc

//Execution instructions
//	$ ./a.out

//NOTE: as part 3 of the assignment didn't specify in which way
//to implement the recursive behaviour, and as I realize that
//part 3 could've been extremely easy just by taking the function
//created from part 2 and adding just an if statement at the
//beginning, combined with a recursive call, I decided to complicate
//it a little bit more, and that's why the code from part 3 is
//a recursive way of calculating (exp(x/b)^b)

//Core functions

//---PART 1---
//function to evaluate the exponential, given accuracy
//if function returns -1, then x was out of range
double approx_exp_PART1(double x, double accur){
  //in case the value is out of range, return -1
  if(x>0.5 || x<0){return -1;}
  if(x == 0.0){
    return 1;
  }
  int i=2, j;
  double y = 1, y_n = y+x, buff;
  //otherwise, return exp (x) up to accuracy accur
  while( y_n-y > accur){
    y = y_n;
    buff = 1;
    //obtaining the correction up to next order
    for(j = 1; j<=i; j++){
      buff *= x/j;
    }
    y_n += buff;
    i++;
  }
  return y_n;
}


//---PART 2---
//evaluate the exponential, given accuracy
//x_b is actually rescaled here as x/b
double approx_exp_PART2(double x_b, int n){
  int i, j;
  double y = 1, buff;
  //otherwise, return exp (x) up to accuracy accur
  for( i=1; i<=n; i++ ){
    buff = 1;
    //obtaining the correction up to next order
    for(j = 1; j<=i; j++){
      buff *= x_b/j;
    }
    y += buff;
  }
  return y;
}

//if function returns -1, then x was out of range
//evaluating the evulation with re-scaling of the exponent
double rescale_approx_exp_PART2(double x, double accur, int b_resc_factor){
  //in case the value is out of range, return -1
  if(x>50 || x<0){return -1;}
  if(x == 0.0){
    return 1;
  }
  //rescaling factor
  cout << "Re-scale factor: " << b_resc_factor << endl;
  //n is the general counter
  int n=2, j;
  //re-scale x
  x = x/b_resc_factor;
  cout << "Value of x rescaled: " << x << endl;
  //y_n_bare is always the approximation without exponentiation
  //(therefore, y_n_bare never represents the solution, at any accuracy)
  double y, y_n = 1+x, y_n_bare;
  //in the case n=1, y_n = (1+x/b)^b, as following:
  y = y_n;
  for(j=0; j<(b_resc_factor-1); j++){
    y_n *= y;
  }
  y = 1;
  //this is the third evaluation of the result, that's why n = 2 above
  while( y_n-y > accur ){
    //set next to the previous
    y = y_n;
    //obtain the approximation without ()^b
    y_n = approx_exp_PART2(x, n);
    y_n_bare = y_n;
    //and now multiply b-1 times
    for(j = 0; j<(b_resc_factor-1); j++){
      y_n *= y_n_bare;
    }
    n++;
  }
  return y_n;
}


//---PART 3---
//
double recursv_part_rescale_PART3(double x_b, double y_n, int n, double accur, int b_resc){
  double y, y_p = y_n, y_p_bare, error_b;
  int j;
  //set next to the previous
  y = y_p;
  //obtain the approximation without ()^b
  y_p = approx_exp_PART2(x_b, n);
  y_p_bare = y_p;
  //and now multiply b-1 times
  for(j = 0; j<(b_resc-1); j++){
    y_p *= y_p_bare;
  }
  //error calculation
  if( y_p-y > 0){error_b = y_p-y;}
  else{error_b = y-y_p;}
  //in case the accuracy is not good enough yet, call recursively
  if( error_b>0.00001 ){
    return recursv_part_rescale_PART3(x_b, y_p, n+1, accur, b_resc);
  }
  else{return y_p;}
}

//if function returns -1, then x was out of range
//evaluating the evulation with re-scaling of the exponent
double rescale_approx_exp_PART3(double x, double accur, int b_resc_factor){
  cout <<"Re-scale factor: " << b_resc_factor << endl;
  //in case the value is out of range, return -1
  if(x>50 || x<-50){return -1;}
  if(x == 0.0){
    return 1;
  }
  //n is the general counter
  double y_n;
  int n=1;
  //re-scale x
  x = x/b_resc_factor;
  cout << "Value of x rescaled: " << x << endl;
  y_n = 1;
  //extracting the result calling the recursive function
  y_n = recursv_part_rescale_PART3(x, y_n, n, accur, b_resc_factor);
  return y_n;
}



//--------------------------------
//main code
int main(){

  //testing part 1
  cout << endl << " ***Testing part 1: " << endl << endl;
  double result_exp;
  double x = 0.3, accuracy = 0.00001;
  result_exp = approx_exp_PART1(x, accuracy);
  if(result_exp == -1){
    cout << "- Value of x out of range." << endl;
  }
  else{
    cout << "- Result of exp(" << x << "): " << result_exp <<
    		", with accuracy: " << accuracy << endl;
  }
  cout << endl;
  x = 11.5;
  result_exp = approx_exp_PART1(x, accuracy);
  if(result_exp == -1){
    cout << "- Value of x out of range." << endl;
  }
  else{
    cout << "- Result of exp(" << x << "): " << result_exp <<
    		", with accuracy: " << accuracy << endl;
  }
  cout << endl;
  x = -23.4;
  result_exp = approx_exp_PART1(x, accuracy);
  if(result_exp == -1){
    cout << "- Value of x out of range." << endl;
  }
  else{
    cout << "- Result of exp(" << x << "): " << result_exp <<
    		", with accuracy: " << accuracy << endl;
  }
  cout << endl;
  
  //testing part 2
  cout << endl << " ***Testing part 2: " << endl << endl;
  //re-scale factor
  int b;
  //obtain, for general case, the value of b (the scale factor)
  x = 0.3;
  b = (int)(2*x) + 1;
  result_exp = rescale_approx_exp_PART2(x, accuracy, b);
  if(result_exp == -1){
    cout << "- Value of x out of range." << endl;
  }
  else{
    cout << "- Result of exp(" << x << "): " << result_exp <<
    		", with accuracy: " << accuracy << endl;
  }
  cout << endl;
  x = 11.5;
  b = (int)(2*x) + 1;
  result_exp = rescale_approx_exp_PART2(x, accuracy, b);
  if(result_exp == -1){
    cout << "- Value of x out of range." << endl;
  }
  else{
    cout << "- Result of exp(" << x << "): " << result_exp <<
    		", with accuracy: " << accuracy << endl;
  }
  cout << endl;
  x = -23.4;
  b = (int)(2*x) + 1;
  result_exp = rescale_approx_exp_PART2(x, accuracy, b);
  if(result_exp == -1){
    cout << "- Value of x out of range." << endl;
  }
  else{
    cout << "- Result of exp(" << x << "): " << result_exp <<
    		", with accuracy: " << accuracy << endl;
  }
  cout << endl;
  
  //testing part 3
  cout << " ***Testing part 3: " << endl << endl;
  //obtain, for general case, the value of b (the scale factor)
  x = 0.3;
  b = (int)(2*x);
  if(b<0){b = -b;}
  b++;
  result_exp = rescale_approx_exp_PART3(x, accuracy, b);
  if(result_exp == -1){
    cout << "- Value of x out of range." << endl;
  }
  else{
    cout << "- Result of exp(" << x << "): " << result_exp <<
    		", with accuracy: " << accuracy << endl;
  }
  cout << endl;
  x = 11.5;
  b = (int)(2*x);
  if(b<0){b = -b;}
  b++;
  result_exp = rescale_approx_exp_PART3(x, accuracy, b);
  if(result_exp == -1){
    cout << "- Value of x out of range." << endl;
  }
  else{
    cout << "- Result of exp(" << x << "): " << result_exp <<
    		", with accuracy: " << accuracy << endl;
  }
  cout << endl;
  x = -23.4;
  b = (int)(2*x);
  if(b<0){b = -b;}
  b++;
  result_exp = rescale_approx_exp_PART3(x, accuracy, b);
  if(result_exp == -1){
    cout << "- Value of x out of range." << endl;
  }
  else{
    cout << "- Result of exp(" << x << "): " << result_exp <<
    		", with accuracy: " << accuracy << endl;
  }
  //end of main
  cout << endl << endl;
  //return 0;
}
