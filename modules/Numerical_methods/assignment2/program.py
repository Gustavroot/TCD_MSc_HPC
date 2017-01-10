#General imports
from math import sin, cos, pi, pow, exp, log10, ceil, log

#Script execution:
#	python program.py

#Importing functions from external files
from bisection_func import bisection_func
from fourier_series import fourier_series
from newton_func import newton_func
from secant_func import secant_func

#LOCAL FUNTIONS: definitions for Fourier expansions
def test_function(w):
    return (1-pow(w, 4))

def test_function_inf_series(x, order_exp):
    approx_v = 0
    approx_v += 3/float(10)

    #and the terms in the summation
    for n in range(1, order_exp+1):
        sub_product = 1*(8/(pow(pi*n, 2)))*((-1+6/(pow(pi*n, 2))))*pow(-1, n)*cos(pi*x*n)
        approx_v += sub_product
    return approx_v

#LOCAL FUNTIONS: definitions for finding roots of functions
def root_find_funct1(x):
    return pow(x, 3)-2

def root_find_funct2(x):
    return exp(x)-2

def root_find_funct1_DER(x):
    return 2*pow(x, 2)

def root_find_funct2_DER(x):
    return exp(x)

#MAIN CODE-----

print "\nAssignment 3 of Num. Methods:\n"

#Expanding functions by Fourier Series
print "------\nFirst part: approximating a function by a Fourier Series expansion: "
order_of_expansion = 10
for w_t in range(1, 6):
    eval_point_t = 1/float(w_t)
    print " --**EVAL. POINT: "+str(eval_point_t)
    print " --Value from original function: "+str(test_function(eval_point_t))
    print " --Finite Fourier approx: "+str(fourier_series(test_function, order_of_expansion*2, eval_point_t))
    print " --Infinite Fourier approx: "+str(2*test_function_inf_series(eval_point_t, order_of_expansion)) #TODO: 2???


print ""
#end of part 1

#--------
#Finding roots of functions
print "------\nSecond part: finding roots in functions.\n"

#List of systems to iterate over
systems_info = list()
systems_info.append([root_find_funct1, 0, 2, pow(10, -6), root_find_funct1_DER, 0, "bisection1.txt"])
systems_info.append([root_find_funct2, 0, 2, pow(10, -6), root_find_funct2_DER, log(2), "bisection2.txt"])

#--
print "Applying bisection method to the two given functions:"
for sys_info in systems_info:
    root_approx = bisection_func(sys_info[0], sys_info[1], sys_info[2], sys_info[3], sys_info[6])
    steps_count = root_approx[1]
    root_approx = root_approx[0]
    print "\troot: "+str(root_approx)
    #expected nr of steps versus really taken
    expect_n = (log10(sys_info[2]-sys_info[1])-log10(sys_info[3]))/log10(2)
    print "\t\texpected n: "+str(int(ceil(expect_n)))
    print "\t\treal n: "+str(steps_count)

#--
print "Applying the secant method to the second function:"
root_approx = secant_func(systems_info[1][0], systems_info[1][1], systems_info[1][2], systems_info[1][3], systems_info[1][5], "secant.txt")
steps_count = root_approx[1]
root_approx = root_approx[0]
print "\troot: "+str(root_approx)
print "\t\treal n: "+str(steps_count)

#--
print "Applying Newton's method to the second function:"
root_approx = newton_func(systems_info[1][0], systems_info[1][4], 0, systems_info[1][3], systems_info[1][5], "newton.txt")
steps_count = root_approx[1]
root_approx = root_approx[0]
print "\troot: "+str(root_approx)
print "\t\treal n: "+str(steps_count)

print ""
#end of part 2
