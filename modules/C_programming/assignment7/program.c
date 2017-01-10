#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

//Compilation instructions:
//	$ gcc program.c -lm

//Execution instructions:
//	$ ./a.out


void merge_arrays(double *A, double *L, int left_count, double *R, int right_count);
void merge_sort_array(double *input_data, int lines_counter);
double *closest_pair_points(char *filename);
double *recursive_closest(double *input_data, int N);
void print_array(double *input_array, int n);
void swap_x_y(double *input_data, int n);


int main(){
  double diff_time;

  //Definition of variables to store times
  struct timeval begin, end;

  //Start timing here
  gettimeofday(&begin, NULL);

  //explanation of the purpose printing msg
  printf("\nProgram for finding the closest pair of points in points.dat\n\n");

  double *out_points;

  //call function to obtain the pair of points
  out_points = closest_pair_points("points.dat");

  //print the pair of points to closest.dat, in only 1 line!
  FILE *f2 = fopen("closest.dat", "w");
  if (f2 == NULL){
    printf("Error opening file!\n");
    exit(1);}
  fprintf(f2, "P1: x = %.10f y = %.10f, P2: x = %.10f y = %.10f\n",
		out_points[0], out_points[1], out_points[2], out_points[3]);
  fclose(f2);
  printf("\n");

  //Stop timing here
  gettimeofday(&end, NULL);

  //Calculating time difference
  diff_time = (end.tv_sec - begin.tv_sec) + 
              ((end.tv_usec - begin.tv_usec)/1000000.0);

  //Printing time to the terminal
  printf("\nExecution time: %.8f.\n\n", diff_time);
  return 0;
}


//if an array 'input_data' is given, such that it has n entries,
//corresponding to n/2 points in a 2D grid, next function prints
//such an array
void print_array(double *input_array, int n){
  int i;
  for(i=0; i<n; i+=2){
    printf("%.2d. x: %.10f, y: %.10f.\n", i/2, input_array[i], input_array[i+1]);
  }
}


//if an array 'input_data' is given, such that it has n entries,
//corresponding to n/2 points in a 2D grid, next function swaps
//each pair of values i.e. 'x' and 'y' in each point
void swap_x_y(double *input_data, int n){
  int i;
  double buff_swapper;
  for(i=0; i<n; i+=2){
    buff_swapper = input_data[i];
    input_data[i] = input_data[i+1];
    input_data[i+1] = buff_swapper;
  }
}


//N is the nr of points
double *recursive_closest(double *input_data, int N){

  //output points
  static double out_points[4];

  //the cases N/2=2 and N/3=3 correspond to the ending conditions of the
  //..recursive calls
  if(N/2==2){
    //for 2 points, there is only one result to return: the 2 points
    out_points[0] = input_data[0]; out_points[1] = input_data[1];
    out_points[2] = input_data[2]; out_points[3] = input_data[3];
    return out_points;
  }
  else if(N/2==3){
    //for 3 points, return the 2 that are the closest
    double dist1, dist2, dist3;
    dist1 = pow(input_data[0]-input_data[2],2)+pow(input_data[1]-input_data[3],2);
    dist1 = sqrt(dist1);
    dist2 = pow(input_data[0]-input_data[4],2)+pow(input_data[1]-input_data[5],2);
    dist2 = sqrt(dist2);
    dist3 = pow(input_data[2]-input_data[4],2)+pow(input_data[3]-input_data[5],2);
    dist3 = sqrt(dist3);
    if(dist1<dist2 && dist1<dist3){
      out_points[0] = input_data[0]; out_points[1] = input_data[1];
      out_points[2] = input_data[2]; out_points[3] = input_data[3];
    }
    else if(dist2<dist1 && dist2<dist3){
      out_points[0] = input_data[0]; out_points[1] = input_data[1];
      out_points[2] = input_data[4]; out_points[3] = input_data[5];
    }
    else{
      out_points[0] = input_data[2]; out_points[1] = input_data[3];
      out_points[2] = input_data[4]; out_points[3] = input_data[5];
    }
    return out_points;
  }
  else{
    int median_location, i, j, buff_array_ctr_l, buff_array_ctr_r;
    double *output_left, *output_right;
    double delta_dist, value_median, delta_dist1, delta_dist2, buff_dist_2;

    //find the median location of input_data with respect to 'x' values
    if((N/2)%2 == 0){
      median_location = N/2;
      value_median = (input_data[median_location]+input_data[median_location-2])/2;
    }
    else{
      //following line means that array_right will always have
      //..less elements than array_left
      median_location = N/2+1;
      value_median = input_data[median_location-2];
    }

    //Splitting data to send in the recursive call
    double *array_left = (double *)malloc(median_location*sizeof(double));
    double *array_right = (double *)malloc((N-median_location)*sizeof(double));
    for(i=0; i<median_location; i++){array_left[i] = input_data[i];}
    for(i=median_location; i<N; i++){array_right[i-median_location] = input_data[i];}

    //this function always returns only 2 points
    output_left = recursive_closest(array_left, median_location);
    output_right = recursive_closest(array_right, N-median_location);

    //***Conquer step: treatment of the median line!***
    //assign to array out_points the minimum pair of points (left or right)
    delta_dist1 = pow(output_left[0]-output_left[2],2)+pow(output_left[1]-output_left[3],2);
    delta_dist1 = sqrt(delta_dist1);
    delta_dist2 = pow(output_right[0]-output_right[2],2)+pow(output_right[1]-output_right[3],2);
    delta_dist2 = sqrt(delta_dist2);
    if(delta_dist1<delta_dist2){
      delta_dist = delta_dist1;
      out_points[0] = output_left[0]; out_points[1] = output_left[1];
      out_points[2] = output_left[2]; out_points[3] = output_left[3];
    }
    else{
      delta_dist = delta_dist2;
      out_points[0] = output_right[0]; out_points[1] = output_right[1];
      out_points[2] = output_right[2]; out_points[3] = output_right[3];
    }

    //LEFT points of the border
    //isolate the elements that live around the median, with a -delta
    buff_array_ctr_l = 0;
    for(i=(median_location-2); i>=0; i-=2){
      if(array_left[i]<(value_median-delta_dist)){break;}
      buff_array_ctr_l++;
    }
    buff_array_ctr_l = buff_array_ctr_l*2;
    //..to isolate the values, create an array with only left values
    double *sub_left = (double *)malloc(buff_array_ctr_l*sizeof(double));
    //..and copying those values
    for(i=0; i<buff_array_ctr_l; i++){
      sub_left[i] = array_left[((median_location-1)-(buff_array_ctr_l-1))+i];
    }
    //'x' and 'y' swap
    swap_x_y(sub_left, buff_array_ctr_l);
    //now that 'y' takes the place of 'y', sub_left can be organized by 'y':
    merge_sort_array(sub_left, buff_array_ctr_l);
    //and then swapped back:
    swap_x_y(sub_left, buff_array_ctr_l);

    //RIGHT points of the border
    //isolate the elements that live around the median, with a +delta
    buff_array_ctr_r = 0;
    for(i=0; i<(N-median_location); i+=2){
      if(array_right[i]<(value_median+delta_dist)){break;}
      buff_array_ctr_r++;
    }
    buff_array_ctr_r = buff_array_ctr_r*2;
    //..to isolate the values, create an array with only right values
    double *sub_right = (double *)malloc(buff_array_ctr_r*sizeof(double));
    //..and copying those values
    for(i=0; i<buff_array_ctr_r; i++){
      sub_right[i] = array_right[i];
    }
    //'x' and 'y' swap
    swap_x_y(sub_right, buff_array_ctr_r);
    //now that 'y' takes the place of 'y', sub_right can be organized by 'y':
    merge_sort_array(sub_right, buff_array_ctr_r);
    //and then swapped back:
    swap_x_y(sub_right, buff_array_ctr_r);

    //BOTH sides of the border: now that both arrays, from right and left of
    //..the border, have been ordered by 'y', iterations can be performed
    //..over both, to determine if within the gap are distances smaller than delta_dist
    for(i=0; i<buff_array_ctr_l; i+=2){
      for(j=0; j<buff_array_ctr_r; j+=2){
        if((array_right[i+1]>(array_left[i+1]-delta_dist)) || (array_right[i+1]<(array_left[i+1]+delta_dist))){
          buff_dist_2 = pow(array_left[i]-array_right[j], 2)+pow(array_left[i+1]-array_right[j+1], 2);
          buff_dist_2 = sqrt(buff_dist_2);
          if(buff_dist_2 < delta_dist){
            out_points[0] = array_left[i]; out_points[1] = array_left[i+1];
            out_points[2] = array_right[j]; out_points[3] = array_right[j+1];
          }
        }
      }
    }

    //free memory:
    free(array_left);
    free(array_right);
    free(sub_left);
    free(sub_right);

    //Finally, returning the 2 closest points of this recursive call
    return out_points;
  }
}


//Function to find and print the closest pair of points
//..that are closest together
//..return the coordinates of each point
double *closest_pair_points(char *filename){
  int lines_counter = 0, i;
  //array to store data
  double *input_data = (double *)malloc(2*100000*sizeof(double));
  //variables for output
  static double out_points[4];
  double *out_buff;
  //Load data from file
  FILE * fp;
  char * line = NULL;
  size_t len = 0;
  ssize_t read;
  //variables for the splitting of the string of each line
  char *pch;
  char x_coord[20], y_coord[20];
  char buff_str[10];
  double x_c, y_c;
  //opening file
  fp = fopen(filename, "r");
  if(fp == NULL)
    exit(EXIT_FAILURE);
  while((read = getline(&line, &len, fp)) != -1){
    //Splitting the string from one line
    //.. x coord
    pch = strtok(line," ");
    strcpy(x_coord, pch);
    x_c = atof(x_coord);
    //.. y coord
    pch = strtok(NULL," ");
    strcpy(y_coord, pch);
    y_c = atof(y_coord);
    //storing data to input_data array
    input_data[2*lines_counter] = x_c;
    input_data[2*lines_counter+1] = y_c;
    lines_counter++;
    if(lines_counter > 100000){
      printf("...number of points n>100000, not allowed.\n\n");
      exit(0);
    }
  }
  //closing file
  fclose(fp);
  if (line)
    free(line);

  //before calling the function to find the closest 2 points,
  //..a sorting of the whole array is made first
  merge_sort_array(input_data, lines_counter*2);

  //Now, with the input_data array sorted by x values, the recursive function
  //..is called
  out_buff = recursive_closest(input_data, lines_counter*2);
  out_points[0] = out_buff[0];
  out_points[1] = out_buff[1];
  out_points[2] = out_buff[2];
  out_points[3] = out_buff[3];

  //free memory
  free(input_data);

  return out_points;
}


//..the 'merge' step in the merge sort algorithm
void merge_arrays(double *A, double *L, int left_count, double *R, int right_count){
  int i,j,k;
  //Indexes for the 3 arrays used in the merging process
  i = 0; j = 0; k =0;
  while(i<left_count && j<right_count) {
    if(L[i] < R[j]){
      A[k] = L[i];
      A[k+1] = L[i+1];
      k+=2; i+=2;
    }
    else{
      A[k] = R[j];
      A[k+1] = R[j+1];
      k+=2; j+=2;
    }
  }
  while(i<left_count){
    A[k] = L[i];
    A[k+1] = L[i+1];
    k+=2; i+=2;
  }
  while(j<right_count){
      A[k] = R[j];
      A[k+1] = R[j+1];
      k+=2; j+=2;
  }
}


//Merge sort algorithm, modified to order by 'x' values
void merge_sort_array(double *A, int n){
  int mid, i;
  double *L, *R;
  //If the array has less than two, elements, do nothing
  if(n < 3) return;
  //Middle point (for splitting), rounded int
  mid = n/2;
  if(mid%2!=0){mid++;}
  //Split into sub-arrays
  L = (double *)malloc(mid*sizeof(double)); 
  R = (double *)malloc((n- mid)*sizeof(double)); 
  //Left sub-array
  for(i=0; i<mid; i++)
    L[i] = A[i];
  //Right sub-array
  for(i = mid; i<n; i++)
    R[i-mid] = A[i];
  //Recursive calls with the sub-arrays
  merge_sort_array(L, mid);
  merge_sort_array(R, n-mid);
  //Merge of sorted sub-arrays
  merge_arrays(A, L, mid, R, n-mid);
  //Release memory used as buffer
  free(L);
  free(R);
}
