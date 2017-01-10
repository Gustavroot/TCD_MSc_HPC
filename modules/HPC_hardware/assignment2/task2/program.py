import subprocess
import os
import sys

#Program to plot matrix multiplication execution times
#and generate a PostScript with the plot of times

#First, is necessary to execute the program for several matrix sizes
#For l, m, n indexes:
#	l will run from 5 to 99
#	n will run from 5 to 99 (a relation l = n will be followed)
#	m will be kept fixed: m = 15

#Array to store time execution values
exec_times = list()

FNULL = open(os.devnull, 'w')
m = 15
for x in range(5, 2000):
    print x
    proc_exec = subprocess.Popen(['./matmul', str(x), str(m), str(x)], stdout=subprocess.PIPE, cwd='../task1/')
    out, err = proc_exec.communicate()
    exec_times.append([str(x)+','+str(m)+','+str(x), float(out.split('\n')[len(out.split('\n'))-3].split(' ')[2])])

#Writing times to output file
file_out = open("execution_times.dat", 'w')
counter_tot = 5
for k in exec_times:
    file_out.write(str(counter_tot)+ " " + str(k[1]) + "\n")
    counter_tot += 1
