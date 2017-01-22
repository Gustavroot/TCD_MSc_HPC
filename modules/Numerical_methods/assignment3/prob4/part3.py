#Program to implement Runge-Kutta, 2nd and 4th orders

#imports
import numpy
from math import log, exp
import matplotlib.pyplot as plt

#CORE functions

#analytical solutions
def analyt_solution1(x):
    return exp(log(0.5)*exp(float(-x)))

#for the following functions, they represent the expressions
#to which is equal the first derivative of y with respect to x
def fnctn1(y,x):
    return -y*log(y)

#Second order Runge-Kutta
def runge_kutta_SecOrder(probe_f, y_0, points_eval):
    #the array 'points_eval' is the set of points over which
    #the function is going to be probed
    n = len( points_eval )
    x = numpy.array( [ y_0 ] * n )
    for i in xrange( n - 1 ):
        h = points_eval[i+1] - points_eval[i]
        k1 = h * probe_f( x[i], points_eval[i] ) / 2.0
        x[i+1] = x[i] + h * probe_f( x[i] + k1, points_eval[i] + h / 2.0 )
    return x

#Fourth order Runge-Kutta
def runge_kutta_FourthOrder(probe_f, y_0, points_eval):
    #the array 'points_eval' is the set of points over which
    #the function is going to be probed
    n = len( points_eval )
    x = numpy.array( [ y_0 ] * n )
    for i in xrange( n - 1 ):
        h = points_eval[i+1] - points_eval[i]
        k1 = h * probe_f( x[i], points_eval[i] )
        k2 = h * probe_f( x[i] + 0.5 * k1, points_eval[i] + 0.5 * h )
        k3 = h * probe_f( x[i] + 0.5 * k2, points_eval[i] + 0.5 * h )
        k4 = h * probe_f( x[i] + k3, points_eval[i+1] )
        x[i+1] = x[i] + ( k1 + 2.0 * ( k2 + k3 ) + k4 ) / 6.0
    return x

#function for plotting data
def plot_f(out_name, title, xlabel, ylabel, x_values, rk2_values, rk4_values, analytic_values):

    plt.title(title)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.plot(x_values, rk2_values, 'b-', label="RK2")
    plt.plot(x_values, rk4_values, 'g--', label="RK4")
    plt.plot(x_values, analytic_values, 'r--', label="Analytic")

    plt.legend(loc=4)
    plt.grid(True)

    plt.savefig(out_name, dpi=None, facecolor='w', edgecolor='w',
		orientation='portrait', papertype=None, format=None,
		transparent=False)



#MAIN code
print "\nFunction to implement 2nd and 4th order Runge-Kutta.\n"

#initial condition
y_0 = 0.5

#grid sizes
grid_val_inv = list()
for i in range(1, 11, 1):
    grid_val_inv.append(pow(2,i))

#creation of output file variable
out_file = open("rungekutta.txt", 'w')

out_rk_2 = list()
out_rk_4 = list()
#calling 2nd order Runge-Kutta
for h_inv in grid_val_inv:
    h = 1.0/(float(h_inv))
    x_points = list()
    x_points.append(0)
    for i in range(h_inv):
        x_points.append(x_points[len(x_points)-1]+h)
    out_rk_2 = runge_kutta_SecOrder(fnctn1, y_0, x_points)
    out_rk_4 = runge_kutta_FourthOrder(fnctn1, y_0, x_points)
    #for this grid value, evaluate the analytical result
    buff_list2 = []
    x = 0
    for i in range(h_inv+1):
        buff_list2.append(analyt_solution1(x))
        x += h
    #determining error in both methods
    rk2_error = 0
    rk4_error = 0
    for index_i, val in enumerate(out_rk_2):
        if abs(val-buff_list2[index_i])>rk2_error: rk2_error = abs(val-buff_list2[index_i])
    for index_i, val in enumerate(out_rk_4):
        if abs(val-buff_list2[index_i])>rk4_error: rk4_error = abs(val-buff_list2[index_i])
    #putting results to output file
    out_file.write(str(h_inv)+"\t"+"RK2 "+str(out_rk_2[len(out_rk_2)-1])+
		"\t"+"RK2 "+str(rk2_error)+"\t"+"RK4 "+str(out_rk_4[len(out_rk_4)-1])
		+"\t"+"RK4 "+str(rk4_error)+"\n")
    #plotting data for case h_inv==4
    if h_inv==4:
        plot_f("comparison.png", "RK2 and RK4 compared to analytical sol.", "x", "y(x)",
		x_points, out_rk_2, out_rk_4, buff_list2)

out_file.close()
