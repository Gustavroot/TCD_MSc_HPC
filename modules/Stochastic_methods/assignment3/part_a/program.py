#General framework to perform integrals over gaussian
#distributions, using Metropolis algorithm

#Execution:
#	$python program.py


#imports
from math import exp
import numpy as np
import matplotlib.pyplot as plt

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
    return np.random.normal(z_0, 1, None)

def proposal_f(z, delta_param):
    #setting the 2nd and 3rd parameters as default
    #first parameter is the mean of the distribution
    #second parameter is Std Dev
    z_prime = np.random.normal(z, 0.5, 1) #if Std Dev is too high (~1), this
                                          #function can loop too many times

    #extra restriction: |x-x'| < delta.. if not, call recursively
    if abs(z-z_prime)>delta_param:
        z_prime = proposal_f(z, delta_param)
    return z_prime

def target_f(z):
    return exp(-pow(float(x),2))


#Function to integrate

def integrand(x):
    return 0




#MAIN code

print "\nFunction implementing integration through Metropolis sampling.\n"

#selection of starting point for the simulation
x_0 = 0

#defining value of parameter 'delta'
delta_p = 0.01

#the states are stored in a dictionary, with 2 keys: {'accepted': [], 'rejected': []},
#where the two values are lists of real points x
states_dict = dict()
states_dict['accepted'] = list()
states_dict['rejected'] = list()

#initial value, from prior distribution, and adding it to the accepted states
x = prior_f(x_0)
states_dict['accepted'].append(x)

#Number of accepted states to be used
nr_states = 5000

#the algorithmic part
buff_counter = 0
while(buff_counter<500):

    #generate proposal state
    x_new = proposal_f(x, delta_p)

    #calculate the acceptance probability
    accept_prob = min([ 1, target_f(x_new)/target_f(x) ])

    #generate flat random number
    u_rand = np.random.uniform()

    #accept or reject
    if u_rand <= accept_prob:
        states_dict['accepted'].append(x_new)
        x = x_new
        #going to next iteration by accepting
        buff_counter += 1
    else:
        states_dict['rejected'].append(x_new)
