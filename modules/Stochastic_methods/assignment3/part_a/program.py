#General framework to perform integrals over gaussian
#distributions, using Metropolis algorithm

#Execution:
#	$python program.py

#NOTE: to see the code implementing integrals using the sampling
#      generated here, go to ../part_b/program.py


#imports
from math import exp
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab

#CORE functions

#The following three are the prior, proposal and target distributions
#specific to this implementation

#  ---prior: Gaussian distribution, centered in the selected x_0
#  ---proposal: Gaussian distribution, centered in the previous state x
#  ---target: exp(-pow(x,2))

#Function implementing the retrieval of values distributed according to
#the prior distribution

def prior_f(z_0):
    #setting the 2nd and 3rd parameters as default
    #first parameter is the mean of the distribution
    return np.random.normal(z_0, 1, 1)

def proposal_f(z, delta_param):
    #setting the 2nd and 3rd parameters as default
    #first parameter is the mean of the distribution
    #second parameter is Std Dev
    z_prime = np.random.normal(z, 1, 1) #if Std Dev is too high (~1), this
                                          #function can loop too many times

    #extra restriction: |x-x'| < delta.. if not, call recursively
    if abs(z-z_prime)>delta_param:
        z_prime = proposal_f(z, delta_param)
    if isinstance(z_prime, list):
        return z_prime[0]
    else:
        return z_prime


#---------
"""
def proposal_f(x, y):
    g1 = mlab.bivariate_normal(x, y, 1.0, 1.0, -1, -1, -0.8)
    g2 = mlab.bivariate_normal(x, y, 1.5, 0.8, 1, 2, 0.6)
    return 0.6*g1+28.4*g2/(0.6+28.4)
"""
#---------


def target_f(z):
    return exp(-pow(z,2))


#Function to integrate

def integrand(z):
    return 0


#Plotting data
def plot_f(out_name, title, xlabel, ylabel, x_values_array, y_values_array):

    plt.title(title)

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    plt.plot(x_values_array, y_values_array, 'b-', label="Target dist.")

    plt.legend(loc=4)
    plt.grid(True)

    plt.savefig(out_name, dpi=None, facecolor='w', edgecolor='w',
		orientation='portrait', papertype=None, format=None,
		transparent=False)




#MAIN code

print "\nFunction implementing integration through Metropolis sampling.\n"

#selection of starting point for the simulation
x_0 = 0

#defining value of parameter 'delta'
delta_p = 10

#the states are stored in a dictionary, with 2 keys: {'accepted': [], 'rejected': []},
#where the two values are lists of real points x
states_dict = dict()
states_dict['accepted'] = list()
states_dict['rejected'] = list()

#initial value, from prior distribution, and adding it to the accepted states
x = prior_f(x_0)
states_dict['accepted'].append(x)

#Number of accepted states to be used
nr_states = 500

#the algorithmic part
buff_counter = 0
while(buff_counter<nr_states):

    #generate proposal state
    x_new = proposal_f(x, delta_p)

    #calculate the acceptance probability
    accept_prob = min([ 1.0, target_f(x_new)/target_f(x) ])

    #generate flat random number
    u_rand = np.random.uniform()

    #accept or reject
    if u_rand <= accept_prob:
        states_dict['accepted'].append(x_new)
        x = x_new
        #going to next iteration by accepting
    else:
        states_dict['rejected'].append(x_new)
    buff_counter += 1

#rejected states
print "Number of rejected states: "+str(len(states_dict['rejected']))
print "Number of accepted states: "+str(len(states_dict['accepted']))


#plotting a histogram to check the generated distribution
#taking the first 100 elements as burn-in
print "Plotting..."

plt.hist(np.asarray(states_dict['accepted']))
plt.title("Target dist. histogram")
plt.xlabel("MC step")
plt.ylabel("State")
plt.savefig("target_dist.png", dpi=None, facecolor='w', edgecolor='w',
	orientation='portrait', papertype=None, format=None,
	transparent=False)
print "...done."


#TODO: uncomment for next part of assignment
#Plotting MC history
#plotting the generated data, to verify distribution
#x_values = range(len(states_dict['accepted']))
#y_values = states_dict['accepted']
#plot_f("target_dist.png", "Target dist.", "MC step", "Y", x_values, y_values)
