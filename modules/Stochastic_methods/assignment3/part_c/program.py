#General framework to perform integrals over gaussian
#distributions, using Metropolis algorithm

#Execution:
#	$python program.py

#NOTE: to see the code implementing integrals using the sampling
#      generated here, go to ../part_b/program.py


#imports
from math import exp, cos, sqrt, pi
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
    z_prime =  z_prime[0]

    #extra restriction: |x-x'| < delta.. if not, call recursively
    if abs(z-z_prime)>delta_param:
        z_prime = proposal_f(z, delta_param)
    return z_prime

def target_f(z):
    return exp(-pow(z,2))/sqrt(pi)


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
    #clearing the plt object, to make it available for next plot
    plt.clf()



#Function performing sampling (Markov Chain part)
#calling the sampler
def sampling_MarkovChain(start_x, delta_prm, nr_sts):
    #the states are stored in a dictionary, with 2 keys: {'accepted': [], 'rejected': []},
    #where the two values are lists of real points x
    states_dict = dict()
    states_dict['accepted'] = list()
    states_dict['rejected'] = list()

    #initial value, from prior distribution, and adding it to the accepted states
    x = prior_f(start_x)
    states_dict['accepted'].append(x)

    #the algorithmic part
    buff_counter = 0
    while(buff_counter<nr_sts):

        #generate proposal state
        x_new = proposal_f(x, delta_prm)

        #calculate the acceptance probability
        accept_prob = min([ 1.0, target_f(x_new)/target_f(x) ])

        #generate flat random number
        u_rand = np.random.uniform()

        #accept or reject
        if u_rand <= accept_prob:
            states_dict['accepted'].append(x_new)
            x = x_new
        else:
            states_dict['rejected'].append(x_new)
        buff_counter += 1

    return states_dict






# *** MAIN code ***

print "\nFunction implementing integration through Metropolis sampling.\n"

#selection of starting point for the simulation
x_0 = 0

#number of states to be used
nr_states = 5000

#values of delta to be used
delta_s = list()
delta_s.append(0.2)
for i in range(500):
    delta_s.append(delta_s[len(delta_s)-1]+0.02)

#obtaining the rejections and acceptances for various deltas
ratios_list = list()
for d_p in delta_s:
    print "Performing calculations for delta = "+str(d_p)+"..."
    out_sampl = sampling_MarkovChain(x_0, d_p, nr_states)
    rat_io = float(len(out_sampl['accepted']))/float(len(out_sampl['accepted'])
		+len(out_sampl['rejected']))
    ratios_list.append(rat_io)
    print "...done."

#plotting data
plot_f("accept_ratio_vs_delta", "Acceptance ratio vs. delta.", "delta", "accept_ratio", delta_s, ratios_list)
