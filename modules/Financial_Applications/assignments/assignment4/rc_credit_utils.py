#CORE functions

#Imports
from math import pow, log, exp, sqrt
from scipy.stats import norm

#AUX functions

#Files processing
#From the output files from ORE execution, return arrays for
#{EEE_b, EE_b, t_k, P, PD, EPE}, and also return the value
#for CVA and EEPE_b
#USAGE: pass the directory name where "Output/" is, and
#pass a list of the parameters to be retrieved, e.g.
#["EEE_b", "t_k"]; possible labels are as in the dict
#'return_values' below
def ore_outputs(dir_name, return_list):

    output = list()

    #Dict with columns to return. The numbers represent the
    #entry in 'observables' list
    return_values = {"EEE_b": 0, "EE_b": 1, "t_k": 2, "P": 3, \
		"PD": 4, "EPE": 5, "EEE": 6, "CVA": 7, "EEPE_b": 8}

    observables = [list(), list(), list(), list(), list(), list(), list()]

    with open(dir_name+"exposure_trade_Swap_20y.csv", 'r') as ore_file:
        #TradeId,Date,Time,EPE,ENE,AllocatedEPE,AllocatedENE,PFE,BaselEE,BaselEEE

        #skip first line of file
        next(ore_file)
        for line in ore_file:
            line_list = line.split(",")
            if float(line_list[8]) == 0:
                break
            observables[0].append(float(line_list[9][:-1]))
            observables[1].append(float(line_list[8]))
            #next line: time
            observables[2].append(float(line_list[2]))
            observables[3].append(float(line_list[3])/float(line_list[8]))
            observables[4].append("?")
            observables[5].append(float(line.split(",")[3]))
            #EEE = EEE_b(t)*P(t)
            observables[6].append(observables[0][len(observables[0])-1]* \
			observables[3][len(observables[3])-1])

        #add extra value for time, for last line
        observables[2].append(float(line_list[2]))

    #extraction of CVA
    with open(dir_name+"xva.csv", 'r') as ore_file:
        next(ore_file)
        next(ore_file)
        cva = next(ore_file)
        cva = float(cva.split(",")[2])
    observables.append(cva)

    eepe_b = 0
    #calculation of baselEEPE
    for index, elem in enumerate(observables[return_values["EEE_b"]]):
        if observables[return_values["t_k"]][index] >= 1:
            break
        eepe_b += elem*(observables[return_values["t_k"]][index+1] - \
		observables[return_values["t_k"]][index])
    eepe_b /= 1#observables[return_values["t_k"]][index-1]
    observables.append(eepe_b)

    for elem in return_list:
        output.append(observables[return_values[elem]])

    #pack returning values with tuple
    return output

#CORE functions

#maturity
def mat():
    #values from ORE output
    ore_outs = ore_outputs("Examples/Example_1/Output/", ["EEE_b", "P", "EE_b", "EEE", "t_k"])
    #factors at numerator
    num1 = 0
    num2 = 0
    #factor at denominator
    den1 = 0
    for index, elem in enumerate(ore_outs[0]):
        if ore_outs[4][index] <= 1:
            num1 += elem*ore_outs[1][index]*(ore_outs[4][index+1]-ore_outs[4][index])
            den1 += elem*ore_outs[3][index]*(ore_outs[4][index+1]-ore_outs[4][index])
        else:
            num2 += elem*ore_outs[2][index]*(ore_outs[4][index+1]-ore_outs[4][index])
    return max(1, min(5, (num1+num2)/den1))

#rho(PD) without adjustment
def rho(pd):
    return 0.24-0.12*(1-exp(-50*pd))

#g(M, PD)
def g_function(pd):
    b = pow(0.11852-0.05478*log(pd), 2)
    #mat() corresponds to maturity
    g = 1+(mat()-2.5)*b
    g /= 1-1.5*b
    return g

#(PHI(...)-PD): middle factor within K(...)
def phi_factor(pd):
    #use of inverse of cumulative normal distribution
    phi_arg = norm.ppf(pd) + sqrt(rho(pd))*norm.ppf(0.999)
    phi_arg /= sqrt(1-rho(pd))
    return norm.cdf(phi_arg) - pd

#LGD
def lgd():
    #TODO: implement, calculate using CVA and reading other info from files
    ore_outs = ore_outputs("Examples/Example_1/Output/", ["CVA", "EPE", "t_k"])
    print
    print ore_outs[0]
    print

    #for 

    return 1.0

#EEPE basel
def eepe_b():
    #As implemented in ore_outputs(...) function from ORE
    #output files
    return ore_outputs("Examples/Example_1/Output/", ["EEPE_b"])[0]

#EEPE basel stressed
def eepe_b_stressed():
    #simplification
    return eepe_b()

#K
def k_function(pd):
    return lgd()*phi_factor(pd)*g_function(pd)

#EAD factor
def ead(alpha):
    return alpha*max(eepe_b(), eepe_b_stressed())

#main RC calculation
def rc_credit(alpha, pd):
    return ead(alpha)*k_function(pd)
