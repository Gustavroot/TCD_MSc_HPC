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


#Functions under consideration for integration

def integrand1(z):
    return cos(z)

def integrand2(z):
    return pow(z, 2)

def trivial_function(z):
    return z



#Functions implementing statistics

#expectation value
def expect_val(samp_data, bin_width, integrand_f):

    out_val = 0
    for i in range(0, len(samp_data), bin_width):
        out_val += integrand_f(samp_data[i])

    return out_val/(len(samp_data)/bin_width)

#variance
def var_val(sampled_data, bin_s, integr_function):

    #new array of binned data
    binned_data = list()
    for i in range(0, len(sampled_data)-bin_s, bin_s):
        binned_data.append(expect_val(sampled_data[i:i+bin_s], 1, trivial_function))

    exp_value = expect_val(binned_data, 1, integr_function)

    out_vari = 0
    for x in binned_data:
        out_vari += pow(integr_function(x)-exp_value, 2)

    return out_vari/(len(binned_data))



# *** MAIN code ***

print "\nProgram to obtain the integrated autocorrelation time.\n"

#selection of starting point for the simulation
x_0 = 0

#number of states to be used
nr_states = 50000

#values of delta to be used
d_p = 0.2

#array of bin sizes
bin_sizes_array = list()
for i in range(1, 1000, 1):
    bin_sizes_array.append(i)
#bin_size = 5

#taking burn-in of 100 elements
out_sampl = sampling_MarkovChain(x_0, d_p, nr_states)['accepted'][100:]

#functions to integrate
functions_list = [integrand1, integrand2]
functions_list_names = ["cosine", "square"]

#looping through the functions under consideration
for i, fnctn in enumerate(functions_list):

    #variance values stored in array, and integr. autoc. times
    variance_vals = list()
    int_autoc_times_array = list()

    #"naive" variance:
    naive_var = var_val(out_sampl, 1, fnctn)

    #generating statistics for different bin sizes
    for bin_size in bin_sizes_array:

        #obtaining the specific value of variance, the "real"
        var_sampl = var_val(out_sampl, bin_size, fnctn)
        variance_vals.append(var_sampl)

        #integrated autocorrelation time
        integr_autoc_time = naive_var/var_sampl
        int_autoc_times_array.append(integr_autoc_time)

        print "---------------"
        print "Bin size: "+str(bin_size)
        print "Integr. autoc. time: "+str(integr_autoc_time)
        print
    #plotting data.. variance vs. bin size
    plot_f("variance_vs_BinSize_"+functions_list_names[i], "Variance vs. bin size: "
		+functions_list_names[i], "bin_size", "variance", bin_sizes_array, variance_vals)
    #plotting integrated autocorrelation times
    plot_f("int_autoc_time_vs_BinSize_"+functions_list_names[i], "Integrated autoc. time vs. bin size: "
		+functions_list_names[i], "bin_size", "int_autoc_time", bin_sizes_array, int_autoc_times_array)
