#Program for estimating integral with the Monte Carlo approach.. uniform distribution

#from subprocess import call
import subprocess
import ast
import math
from math import sqrt

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
def monte_carlo_integration(funct_to_average, N):
    #Parsing the mediator_ranlux program's output.. to a Python list
    rand_nrs = ast.literal_eval(ranlux_call(N))
    #Summing over the N random samples
    sum_of_rands = 0
    for k in rand_nrs:
        sum_of_rands += funct_to_average(k)
    #Averaging (i.e. integrating)
    average_result = (sum_of_rands)/N
    #Summing over random samples again, but for variances this time
    sum_of_rands_approx = 0
    for k in rand_nrs:
        sum_of_rands_approx += (funct_to_average(k)-average_result)**2
    #Returning: average and standard deviation
    return (average_result, sqrt(sum_of_rands_approx/N))

#Functions for which averages are calculated using uniformly distributed
#random numbers
def funct1(x):
    val_to_return = 1
    val_to_return *= 3
    val_to_return *= math.exp(-3*x)
    val_to_return *= 1/(1+x/3)
    return val_to_return

#-----------

print "Program for the approximation of an integral with Monte Carlo methods."
print "\n"

print monte_carlo_integration(funct1, 100)
print monte_carlo_integration(funct1, 1000)
print monte_carlo_integration(funct1, 10000)
print monte_carlo_integration(funct1, 100000)
