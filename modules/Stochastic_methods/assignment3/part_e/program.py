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

    return states_dict


#Function to average data
def simple_expt_value(list_values):
    result_o = 0
    for i in list_values:
        result_o += i
    return result_o/len(list_values)


#Function to bin data
def bin_data(dat_smpld, bin_size):
    #new array of binned data
    binned_data = list()
    for i in range(0, len(dat_smpld)-bin_size, bin_size):
        binned_data.append(dat_smpld[i+int(bin_size/2)]) #simple_expt_value(dat_smpld[i:i+bin_size]))
    return binned_data


#Function integrating (Monte Carlo part)
def integrate_MonteCarlo(initial_x, d_param, nr_points, integrand, plot_name, b_size):
    #generate samples
    out_sampling = sampling_MarkovChain(initial_x, d_param, nr_points, plot_name)

    #taking a burn-in of 100
    out_sampling = out_sampling['accepted'][100:]

    #before integrating: bin data
    out_sampling = bin_data(out_sampling, b_size)

    out_result = 0
    #integrate using Monte Carlo
    for x_i in out_sampling:
        out_result += integrand(x_i)
    out_result = out_result/len(out_sampling)

    #multiplying for a factor of 1/sqrt(pi)
    out_result = out_result

    return out_result

#MC integration without Markov Chain sampling
def plain_MC_int(state_values, integrnd_f):
    result_p_MC = 0
    for x in state_values:
        result_p_MC += integrnd_f(x)

    return result_p_MC/len(state_values)




# *** MAIN code ***

print "\nFunction implementing integration through Metropolis sampling.\n"

#selection of starting point for the simulation
x_0 = 0

#number of states to be used
nr_states = 50000

#values of delta to be used
delta_s = [0.2]

bin_s = 30

#array of functions to be integrated
fs_to_int = [integrand1, integrand2]
fs_to_int_names = ["cosine", "x^2"]
fs_to_int_explicit_names = ["cosine", "square"]
exact_result_list = [0.7788007831, 0.5]
values_variances = [0.065, 0.4]

#integrate both functions for the three values of delta
for d_val in delta_s:
    print "DELTA = "+str(d_val)
    print "*********"
    for i, integrnd in enumerate(fs_to_int):
        name_for_plot = "MC_history_D_"+"{0:.2f}".format(d_val)+"_"+fs_to_int_explicit_names[i]
        output_MC_int = integrate_MonteCarlo(x_0, d_val, nr_states, integrnd, name_for_plot, bin_s)
        print " -- function integrated: "+fs_to_int_names[i]
        print "\tresult: "+str(output_MC_int)
        print "\terror: "+str(abs(output_MC_int-exact_result_list[i]))
    print


#Now, alternatively, using n normal random dist.s to integrate
#structure of the array: [n_cosine, n_square]
n_s = [20833, 19639]

print "\nIntegration through n statistically independent Gaussian:"

#integrating using the n indep. Gaussians.. values taken from Python
#function to generate random normal numbers
for indx, val in enumerate(n_s):
    stt_values =  np.random.normal(x_0, 1.0/sqrt(2.0), val)
    result = plain_MC_int(stt_values, fs_to_int[indx])
    print " -- result for "+fs_to_int_names[indx]+": "+str(result)
    print " -- error for "+fs_to_int_names[indx]+": "+str(abs(result-exact_result_list[indx]))
