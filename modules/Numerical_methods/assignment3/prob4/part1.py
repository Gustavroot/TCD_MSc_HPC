#Program to implement Euler's method in 3 different ODEs

#imports
from math import exp, log

#CORE functions

#analytical solutions
def analyt_solution1(x):
    return exp(log(0.5)*exp(float(-x)))

def analyt_solution2(x):
    return 1.0/4.0+(3.0/4.0)*exp(-4.0*x)

def analyt_solution3(x):
    return exp(float(x))

#for the following functions, they represent the expressions
#to which is equal the first derivative of y with respect to x
def fnctn1(x,y):
    return -y*log(y)

def fnctn2(x,y):
    return 1-4*y

def fnctn3(x,y):
    return y

#function implementing Euler's method
#probe_fnctn is the the expression for dy/dx
def euler_method(grid_val_inv, probe_functn, initial_val):
    output_list = list()
    y = initial_val
    x = 0
    h = 1.0/(float(grid_val_inv))
    for i in range(grid_val_inv):
        output_list.append(y)
        y += probe_functn(x,y)*h
        x += h
    output_list.append(y)
    return output_list


#MAIN code

print "\nFunction to implement Euler's method for ODEs.\n"

#array of functions:
#expressions for derivatives
array_fncnts_der = [fnctn1, fnctn2, fnctn3]
#analytical solutions
array_fncnts_analyt = [analyt_solution1, analyt_solution2, analyt_solution3]
#initial values for the corresponding functions
init_values = [0.5, 1, 1]

#grid sizes
grid_sizes_inverse = list()
for i in range(1, 11, 1):
    grid_sizes_inverse.append(pow(2,i))

#for each value in grid_size_inverse, find an array of
#numerical solution, and then compare that array with values
#obtained from the analytical result

#'index_f' represents the index of that function, within the 3
#possible functions evaluated with this program

#buff_list1 is for approx, and buff_list2 is for analytical
buff_list1 = list()

files_list = [open("forward_1.txt", 'w'), open("forward_2.txt", 'w'), open("forward_3.txt", 'w')]
for index_f, fncnt_p in enumerate(array_fncnts_der):
    for h_inv in grid_sizes_inverse:
        h = 1.0/(float(h_inv))
        buff_list1 = euler_method(h_inv, fncnt_p, init_values[index_f])
        #for this grid value, evaluate the analytical result
        buff_list2 = []
        x = 0
        for i in range(h_inv+1):
            buff_list2.append(array_fncnts_analyt[index_f](x))
            x += h
        #obtaining the error, defined as max difference between values
        max_val = 0
        for index_i, val in enumerate(buff_list1):
            if abs(val-buff_list2[index_i])>max_val: max_val = abs(val-buff_list2[index_i])
        files_list[index_f].write(str(h_inv)+"\t"+str(buff_list1[h_inv])+"\t"+str(max_val)+"\n")

for file_x in files_list:
    file_x.close()
