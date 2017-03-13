#Program to implement Euler's method in 3 different ODEs

#imports
import numpy as np
import matplotlib.pyplot as plt
from math import exp, log, sqrt

#CORE functions

#function implementing Euler-Maruyama method
#probe_fnctn is the the expression for dy/dx
def euler_maruyama_method(a, b, rwalk, h):
    output_list = list()
    
    #setting initial values for X(t) and t
    y = 1.0
    t = 0
    w_prev = 0
    
    for W_n in rwalk:
        output_list.append(y)
        y += a*h + b*(W_n - w_prev)
        w_prev = W_n
        t += h
    output_list.append(y)
    return output_list


#MAIN code

print "\nFunction to implement Euler-Maruyama method for SDEs.\n"

#grid sizes - 2, 4, 8, 16, 32
nsteps = list()
for i in range(1, 6, 1):
    nsteps.append(pow(2,i))

#buff_list1 is for approx, and buff_list2 is for analytical
buff_list1 = list()
errors = list()

#parameters of the SDE
a = 1.5
b = 1

#up to time:
y = 2

errs_strong = list()
errs_weak = list()

for n in nsteps:
    h = 1.0/(float(n))
    
    #creation of 'x' axis (with re-scaling)
    t = np.arange(1, nsteps[len(nsteps)-1]*y+1, int( nsteps[len(nsteps)-1] / n ))
    steps = np.random.randint(0, 2, size = n*y)
    steps = np.where(steps > 0, 1, -1)
    steps = steps*(1/sqrt(n))
    rwalk = steps.cumsum()

    buff_list1 = euler_maruyama_method(a, b, rwalk, h)
    
    #adjusting output to analytic solution X(t)
    buff_list2 = list()
    buff_list2.append(1)
    for indx, z in enumerate(rwalk):
        buff_list2.append(exp(t[indx]/n+z))
    
    #at this point, 'buff_list1' contains data for the numerical
    #solution, and 'buff_list2' for the analytic solution

    #STRONG convergence
    error_strong = 0
    for indx, elem in enumerate(buff_list1):
        error_strong += abs(elem - buff_list2[indx])
    error_strong /= len(buff_list1)
    
    #WEAK convergence
    error_weak_1 = 0
    for elem in buff_list1:
        error_weak_1 += elem
    error_weak_1 /= len(buff_list1)
    error_weak_2 = 0
    for elem in buff_list2:
        error_weak_2 += elem
    error_weak_2 /= len(buff_list2)
    error_weak = abs(error_weak_1 - error_weak_2)
    
    errs_strong.append(error_strong)
    errs_weak.append(error_weak)

print ""

#plotting strong errors
sizes_log = list()
for n in nsteps:
    sizes_log.append(log(n))
errors_log = list()
for err in errs_strong:
    errors_log.append(log(err))
plt.title("log-log strong errors plot")
plt.plot(sizes_log, errors_log)
plt.savefig("./log-log-strong-errors.png", dpi=None, facecolor='w', edgecolor='w',
	orientation='portrait', papertype=None, format=None,
	transparent=False)

plt.clf()
#plotting strong errors
sizes_log = list()
for n in nsteps:
    sizes_log.append(log(n))
errors_log = list()
for err in errs_weak:
    errors_log.append(log(err))
plt.title("log-log weak errors plot")
plt.plot(sizes_log, errors_log)
plt.savefig("./log-log-weak-errors.png", dpi=None, facecolor='w', edgecolor='w',
	orientation='portrait', papertype=None, format=None,
	transparent=False)
