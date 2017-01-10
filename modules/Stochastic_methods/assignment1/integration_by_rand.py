#Program for estimating integral with the Monte Carlo approach

#from subprocess import call
import subprocess
import ast
import datetime

#Function to call the RANLUX random numbers generator
#IMPORTANT: the generated random numbers live in the interval
#including 0 and excluding 1. N is the cardinality of the
#set of random numbers to retrieve from RANLUX
def ranlux_call(N):
    #The mediator_ranlux program is executed, and its output PIPEd
    proc = subprocess.Popen(['./mediator_ranlux', str(N)], stdout=subprocess.PIPE)
    #The output of mediator_ranlux is taken and returned
    output = proc.stdout.read()
    return output

#General function for integrating by Monte Carlo
#..where N is the number of random numbers generated
#..type_of_average can be 'exact' or 'approx'
def monte_carlo_integration(funct_to_average, N, label_function):
    #For 'exact' use of average (mu), the analytic values are used
    averages = {'linear': 0.5, 'quadratic': 1/3.0, 'sqrt2': 2/3.0}
    #Parsing the mediator_ranlux program's output.. to a Python list
    rand_nrs = ast.literal_eval(ranlux_call(N))
    #Summing over the N random samples
    sum_of_rands = 0
    for k in rand_nrs:
        sum_of_rands += funct_to_average(k)
    #Averaging (i.e. integrating)
    average_result = (sum_of_rands)/N
    #Summing over random samples again, but for variances this time
    sum_of_rands_exact = 0
    sum_of_rands_approx = 0
    for k in rand_nrs:
        sum_of_rands_exact += (funct_to_average(k)-averages[label_function])**2
        sum_of_rands_approx += (funct_to_average(k)-average_result)**2
    #Returning a Python tuple: (average, variance with exact mu, variance with approx mu)
    return (average_result, sum_of_rands_exact/N, sum_of_rands_approx/N)

#Functions for which averages are calculated using uniformly distributed
#random numbers
def funct1(x):
    return x

def funct2(x):
    return x*x

def funct3(x):
    return x**(1/2.0)

#Function for generating data of Average vs. N
#  a .dat file is exported for the 3 types of values returned
#  by the function monte_carlo_integration, i.e. contained in the
#  tuple
def average_running(funct_for_avrg_iter, label_function):
    #Dict to associate exact average
    averages = {'linear': 0.5, 'quadratic': 1/3.0, 'sqrt2': 2/3.0}
    #Four files are created, as the error is going to be plotted as well
    tmp_file = open('./data_for_plots/'+label_function+".dat", 'w')
    tmp_file_errors = open('./data_for_plots/'+"errors"+label_function+".dat", 'w')
    tmp_file_variances_exact = open('./data_for_plots/'+"exactvariances"+label_function+".dat", 'w')
    tmp_file_variances_approx = open('./data_for_plots/'+"approxvariances"+label_function+".dat", 'w')
    #Performing the integration up to 350 000 samples, starting with 100 samples,
    #but in jumps of 100. i.e. 100, 200, 300, 400, ..., 350 000
    for k in range(100, 350000, 100):
        #The following line performs the whole num simulation
        monte_carlo_approx = monte_carlo_integration(funct_for_avrg_iter, k, label_function)
        #Writing to files
        tmp_file.write(str(k)+"\t"+str(monte_carlo_approx[0])+"\n")
        tmp_file_errors.write(str(k)+"\t"+str(monte_carlo_approx[0]-averages[label_function])+"\n")
        tmp_file_variances_exact.write(str(k)+"\t"+str(monte_carlo_approx[1])+"\n")
        tmp_file_variances_approx.write(str(k)+"\t"+str(monte_carlo_approx[2])+"\n")
        print "Finished iteration "+str(k)+" in function "+label_function+"..."
    #Closing open files after writing
    tmp_file.close()
    tmp_file_errors.close()
    tmp_file_variances_exact.close()
    tmp_file_variances_approx.close()

#-----------

print "Program for the approximation of integrals with Monte Carlo methods."
print "\n"
#The following is the list of functions to average
#functions_2_aver = [funct1, funct2, funct3]

#Next, is necessary to iterate both on N and on the 3 given functions
print "Starting approx for integration of linear..."
print "...starting time: "+str(datetime.datetime.now().time())
average_running(funct1, "linear")
print "...done with linear."
print "Starting approx for integration of quadratic..."
print "...starting time: "+str(datetime.datetime.now().time())
average_running(funct2, "quadratic")
print "...done with quadratic."
print "Starting approx for integration of sqrt..."
print "...starting time: "+str(datetime.datetime.now().time())
average_running(funct3, "sqrt2")
print "...done with sqrt."
