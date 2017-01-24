#General framework to perform integrals over gaussian
#distributions, using Metropolis algorithm

#Execution:
#	$python program.py


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


#Functions to integrate

def integrand1(z):
    return cos(z)

def integrand2(z):
    return pow(z, 2)


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
def sampling_MarkovChain(start_x, delta_prm, nr_sts, png_out_name):
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

    #Plotting MC history
    #plotting the generated data, to verify distribution
    #removing burn-in of 100 elements
    x_values = range(len(states_dict['accepted'][100:]))
    y_values = states_dict['accepted'][100:]
    plot_f(png_out_name+".png", "Target dist. DELTA = "+str(delta_prm), "MC step", "Y", x_values, y_values)

    #Plotting histogram
    #removing burn-in of 100 elements
    plt.hist(np.asarray(states_dict['accepted'][100:]))
    plt.title("Target dist. histogram")
    plt.xlabel("MC step")
    plt.ylabel("State")
    plt.savefig("HISTOGRAM_"+png_out_name+".png", dpi=None, facecolor='w', edgecolor='w',
		orientation='portrait', papertype=None, format=None,
		transparent=False)
    plt.clf()

    return states_dict


#Function integrating (Monte Carlo part)
def integrate_MonteCarlo(initial_x, d_param, nr_points, integrand, plot_name):
    #generate samples
    out_sampling = sampling_MarkovChain(initial_x, d_param, nr_points, plot_name)

    out_result = 0
    #integrate using Monte Carlo
    #taking a burn-in of 100
    out_sampling = out_sampling['accepted'][100:]
    for x_i in out_sampling:
        out_result += integrand(x_i)
    out_result = out_result/len(out_sampling)

    #multiplying for a factor of 1/sqrt(pi)
    out_result = out_result

    return out_result




# *** MAIN code ***

print "\nFunction implementing integration through Metropolis sampling.\n"

#selection of starting point for the simulation
x_0 = 0

#number of states to be used
nr_states = 50000

#values of delta to be used
delta_s = [0.3, 0.7, 1.1]

#array of functions to be integrated
fs_to_int = [integrand1, integrand2]
fs_to_int_names = ["cosine", "x^2"]
fs_to_int_explicit_names = ["cosine", "square"]

#integrate both functions for the three values of delta
for d_val in delta_s:
    print "DELTA = "+str(d_val)
    print "*********"
    for i, integrnd in enumerate(fs_to_int):
        name_for_plot = "MC_history_D_"+"{0:.2f}".format(d_val)+"_"+fs_to_int_explicit_names[i]
        output_MC_int = integrate_MonteCarlo(x_0, d_val, nr_states, integrnd, name_for_plot)
        print " -- function integrated: "+fs_to_int_names[i]
        print "\tresult: "+str(output_MC_int)
    print
