#Implementation of natural spline approximation to Runge function

#Following library is for matrix implementations
import numpy
from numpy.linalg import inv
import matplotlib.pyplot as plt
import os
import time
import subprocess

#Execution:
#	python natural_spline.py

#--


#Definition of Runge function
def runge_function(y_var):
    return 1/(1+25*y_var*y_var)


#Evalutation of derivative of Runge function
def runge_function_derivative(y_var):
    return (-50*y_var)/((1+25*y_var*y_var)*(1+25*y_var*y_var))


#Spline approximation
#This function receives the following parameters
#	-- range of horizontal values: x_min, x_max
#	-- grid splitting: h
#	-- values for fitting: array_y_values
def spline_approximation(spline_type, x_min, x_max, h, array_y_values):
    #As explained in eq. 3.1 in ./spline_interp_description.pdf,
    #there are m+1 points, where:
    m = (x_max-x_min)/h

    #Depending on the type of spline, boundary conditions are set for
    #the second derivative of the splines
    second_deriv_spline_vals = list()
    #This if-elif is the only difference between the two splines
    if spline_type == 'natural':
        second_deriv_spline_vals.append(0)
        second_deriv_spline_vals.append(0)
    elif spline_type == 'natural': #TODO: modify this to generalize more
        second_deriv_spline_vals.append(runge_function(x_min))
        second_deriv_spline_vals.append(runge_function(x_max))

    #First, definition of the matrix to invert (is the same in both cases)
    #For definition of the matrix, a list of lists is necessary:
    list_const_matrix = list()
    #..initial row
    list_const_matrix.append(list())
    list_const_matrix[0].append(4.0)
    list_const_matrix[0].append(1.0)
    for i in range(0, int(m-3)):
        list_const_matrix[0].append(0.0)
    #..middle rows
    i = 0
    while i<m-3:
        list_const_matrix.append(list())
        for k in range(0, i):
            #fill left zeros
            list_const_matrix[i+1].append(0.0)
        list_const_matrix[i+1].append(1.0)
        list_const_matrix[i+1].append(4.0)
        list_const_matrix[i+1].append(1.0)
        for k in range(i+3, int(m-1)):
            #fill right zeros
            list_const_matrix[i+1].append(0.0)
        i += 1
    #..final row
    list_const_matrix.append(list())
    for k in range(0, int(m-3)):
        list_const_matrix[i+1].append(0.0)
    list_const_matrix[i+1].append(1.0)
    list_const_matrix[i+1].append(4.0)
    #If 'complete', an expansion and modification should be made:
    if spline_type == 'complete':
        list_const_matrix_expanded = list()
        #fill with zeros
        for i in range(0, int(m+1)):
            list_const_matrix_expanded.append(list())
            for j in range(0, int(m+1)):
                list_const_matrix_expanded[i].append(0)
        #the inner matrix, with no external layer, is almost equal to the original
        for i in range(0, int(m-1)):
            for j in range(0, int(m-1)):
                list_const_matrix_expanded[i][j+1] = list_const_matrix[i][j]
        #and then, it's necessary an specification of the external layer
        #..left wall
        list_const_matrix_expanded[0][0] = 1
        for i in range(1, int(m+1)):
            list_const_matrix_expanded[i][0] = 0
        #..right wall
        for i in range(0, int(m-1)):
            list_const_matrix_expanded[i][int(m)] = 0
        list_const_matrix_expanded[int(m-2)][int(m)] = 1
        list_const_matrix_expanded[int(m-1)][int(m)] = 1
        list_const_matrix_expanded[int(m)][int(m)] = 1
        #..rest of walls from below
        list_const_matrix_expanded[int(m-1)][1] = 2
        for i in range(2, int(m-1)):
            list_const_matrix_expanded[int(m-1)][i] = 2
        list_const_matrix_expanded[int(m)][int(m-1)] = 2
        for i in range(1, int(m-2)):
            list_const_matrix_expanded[int(m)][i] = 2
        list_const_matrix = list_const_matrix_expanded

    #Then, we proceed to instance the matrix
    const_matrix = numpy.matrix(list_const_matrix)

    #Debugging:
    """
    if int(m) == 32:
        print const_matrix
    """

    #Creating now the column vector at right side of equality
    righ_equal_column = list()
    #..first element
    righ_equal_column.append([array_y_values[2]- \
        2*array_y_values[1]+array_y_values[0]])
    #..middle elements
    for k in range(1, int(m-2)):
        righ_equal_column.append([array_y_values[int(k+1)]-
            2*array_y_values[int(k)]+array_y_values[int(k-1)]])
    #..last element
    righ_equal_column.append([array_y_values[int(m)]- \
        2*array_y_values[int(m)-1]+array_y_values[int(m)-2]])
    #And expanding in case of 'complete'
    if spline_type == 'complete':
        righ_equal_column.append([(6/h)*((array_y_values[3]-array_y_values[1])/(h)-(runge_function_derivative(x_min+h)))])
        righ_equal_column.append([(6/h)*((array_y_values[int(m)]-array_y_values[int(m-1)])/(h)-(runge_function_derivative(x_min+int(m)*h)))])
    #And converting this into a matrix
    ys_column_vector = numpy.matrix(righ_equal_column)

    #Vector of solutions for the sigmas:
    #..inverse of left-side matrix
    inv_matrix = inv(const_matrix)
    #print inv_matrix
    #..multiplication of both, i.e. column solution for sigmas
    sigmas_column_solution = inv_matrix.dot(ys_column_vector)
    #..and finally, adding sigma_0 and sigma_m to the solution vector
    #if 'natural'
    final_vector_sigmas = list()
    if spline_type == 'natural':
        final_vector_sigmas = [0]
        for sigma_v in list(sigmas_column_solution):
            final_vector_sigmas.append(sigma_v[0,0])
        final_vector_sigmas.append(0)

    elif spline_type == 'complete':
        for sigma_v in list(sigmas_column_solution):
            final_vector_sigmas.append(sigma_v[0,0])
        

    #Debugging:
    """
    print final_vector_sigmas
    """

    """
    #In case of 'complete', it's necessary a new reduction of the sigmas vector
    if spline_type == 'complete':
        list_reduction_matrix = list()
        for i in range(1, int(m-1)):
            list_reduction_matrix = final_vector_sigmas[i]
        final_vector_sigmas = numpy.matrix(list_reduction_matrix)
    """

    #And now, the necessary 4m parameters are obtained
    parameters_list = list()
    for k in range(0, int(m)):
        parameters_list.append(list())
        #a_i
        buff_val = (final_vector_sigmas[k+1]-final_vector_sigmas[k])/(6*h)
        parameters_list[len(parameters_list)-1].append(buff_val)
        #b_i
        buff_val = final_vector_sigmas[k]/2
        parameters_list[len(parameters_list)-1].append(buff_val)
        #c_i
        buff_val = (array_y_values[k+1]-array_y_values[k])/h
        buff_val -= h*((2*final_vector_sigmas[k]+final_vector_sigmas[k+1])/6)
        parameters_list[len(parameters_list)-1].append(buff_val)
        #d_i
        buff_val = array_y_values[k]
        parameters_list[len(parameters_list)-1].append(buff_val)

    return parameters_list


#Function to generate the points to pass over to spline_approximation()
#function
#This function returns y values, an array
def points_eval(function_to_eval, x_min, x_max, h):
    #Array to store evaluation points to pass later to spline approx
    eval_points_array = list()
    for i in range(0, int((x_max-x_min)/h)+1):
        eval_points_array.append(function_to_eval(x_min+i*h))
    return eval_points_array


#Plotting function
#Two options for plotting with gnuplot here:
#	install and use http://gnuplot-py.sourceforge.net/
#	call gnuplot command from Python
#..the second option is taken, as is better in case of
#  cross-platform works or executions
def plot_cubic_spline(type_spline, array_params, x_min, x_max, h):
    print "...generating data for spline..."

    m = int((x_max-x_min)/h)

    #Plotting with gnuplot:
    #..creating the gnuplot file dynamically from here
    #..FIRST: for the approximation functions
    out_gnufile_str = "set xrange ["+str(x_min)+":"+str(x_max)+"]\n"
    out_gnufile_str += "set samples 1000\n"

    #..dynamic function definition here
    out_gnufile_str += "funct_s(x) = "
    buff_spline_str_form = ""
    for k in range(1, m+1):
        buff_spline_str_form += str(array_params[k-1][0])+"*"
        buff_spline_str_form += "(x-"+str(x_min+(k-1)*h)+")**3+"
        buff_spline_str_form += str(array_params[k-1][1])+"*"
        buff_spline_str_form += "(x-"+str(x_min+(k-1)*h)+")**2+"
        buff_spline_str_form += str(array_params[k-1][2])+"*"
        buff_spline_str_form += "(x-"+str(x_min+(k-1)*h)+")+"
        buff_spline_str_form += str(array_params[k-1][3])
        out_gnufile_str += "x>"+str(x_min+(k-1)*h)+" && x<" + \
            str(x_min+k*h) + " ? "+ buff_spline_str_form + " : "
        buff_spline_str_form = ""
    out_gnufile_str += "0.5\n"

    #..extra gnuplot settings
    out_gnufile_str += "set style line 1 linecolor rgb '#0060ad'\n"
    out_gnufile_str += "plot funct_s(x) with lines linestyle 1\n"
    out_gnufile_str += "set term png\n"
    out_gnufile_str += "set output '"+"plots/SplineFit_"+type_spline+"_h"+str(h)+".png"+"'\n"
    out_gnufile_str += "replot\n"

    #Debugging:
    #print out_gnufile_str

    #Write this str to gp file
    output_gp_file = open("data_params/SplineFit_"+type_spline+"_h"+str(h)+".dat", 'w')
    output_gp_file.write(out_gnufile_str)

    #..name of saved file in following line
    print "...done, data saved as: "+"data_params/SplineFit_"+type_spline+"_h"+str(h)+".dat"

    #..SECOND: for the error functions
    out_gnufile_str = "set xrange ["+str(x_min)+":"+str(x_max)+"]\n"
    out_gnufile_str += "set samples 1000\n"

    #..dynamic function definition here
    out_gnufile_str += "funct_s(x) = "
    buff_spline_str_form = ""
    for k in range(1, m+1):
        buff_spline_str_form += str(array_params[k-1][0])+"*"
        buff_spline_str_form += "(x-"+str(x_min+(k-1)*h)+")**3+"
        buff_spline_str_form += str(array_params[k-1][1])+"*"
        buff_spline_str_form += "(x-"+str(x_min+(k-1)*h)+")**2+"
        buff_spline_str_form += str(array_params[k-1][2])+"*"
        buff_spline_str_form += "(x-"+str(x_min+(k-1)*h)+")+"
        buff_spline_str_form += str(array_params[k-1][3])
        #subtracting now the approx function to the original
        buff_spline_str_form = "1/(1+25*x*x)"+"-("+buff_spline_str_form+")"
        out_gnufile_str += "x>"+str(x_min+(k-1)*h)+" && x<" + \
            str(x_min+k*h) + " ? "+ buff_spline_str_form + " : "
        buff_spline_str_form = ""
    out_gnufile_str += "0.5\n"

    #..extra gnuplot settings
    out_gnufile_str += "set style line 1 linecolor rgb '#0060ad'\n"
    out_gnufile_str += "plot funct_s(x) with lines linestyle 1\n"
    out_gnufile_str += "set term png\n"
    out_gnufile_str += "set output '"+"plots/ErrorData_"+type_spline+"_h"+str(h)+".png"+"'\n"
    out_gnufile_str += "replot\n"

    #Debugging:
    #print out_gnufile_str

    #Write this str to gp file
    output_gp_file_errors = open("data_params/ErrorData_"+type_spline+"_h"+str(h)+".dat", 'w')
    output_gp_file_errors.write(out_gnufile_str)

    #..name of saved file in following line
    print "...and now the errors: data saved as: "+"data_params/ErrorData_"+type_spline+"_h"+str(h)+".dat"


#---MAIN PROGRAM
#---------------

print ""
print "Program to approximate a function with cubic splines:\n"

grid_points = [2.0, 4.0, 8.0, 16.0]

#Range of values over horizontal line
x_min = -1.0
x_max = 1.0

#Arrays are necessary for storing the parameters resulting
#from the spline approximation

print "-----"
print "First, natural spline:"
#First, natural apline approximation
natural_spline_parameters = list()
for h_inv in grid_points:
    array_y_values = points_eval(runge_function, x_min, x_max, 1/h_inv)
    natural_spline_parameters = \
        spline_approximation('natural', x_min, x_max, 1/h_inv, array_y_values)
    #And finally, plot the results
    plot_cubic_spline('natural', natural_spline_parameters, x_min, x_max, 1/h_inv)

print "\n------\nSecond, complete spline:"
#Second, complete apline approximation
complete_spline_parameters = list()
for h_inv in grid_points:
    array_y_values = points_eval(runge_function, x_min, x_max, 1/h_inv)
    complete_spline_parameters = \
        spline_approximation('complete', x_min, x_max, 1/h_inv, array_y_values)
    #And finally, plot the results
    plot_cubic_spline('complete', complete_spline_parameters, x_min, x_max, 1/h_inv)

print ""
