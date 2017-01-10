#Program for estimating integral with the Monte Carlo approach
import pyranlux

#Function to call the RANLUX random numbers generator
#IMPORTANT: the generated random numbers live in the interval
#including 0 and excluding 1. N is the cardinality of the
#set of random numbers to retrieve from RANLUX
#Source for wrapping:
#	http://www.scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html
def ranlux_call(N):
    return pyranlux.genrand(N)

#	python setup.py build_ext --inplace
#	python integrals_estimation.py

#Function to calculate a integral using uniformly distributed
#random numbers


#-----------

ranlux_output = ranlux_call(7)

print ranlux_output
