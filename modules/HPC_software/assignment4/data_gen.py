#Data generator for testing OpenMP performance
#improvements

import subprocess
import os.path
import os
import sys

#Execution instructions:
#	$ python data_gen.py


if __name__ == "__main__":

    #if executables don't exist, create them
    if not os.path.isfile("./task1/a.out"):
        cmd = ['gcc', '-fopenmp', './task1/gauss.c', '-o', './task1/a.out']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p.wait()
    if not os.path.isfile("./task2/a.out"):
        cmd = ['gcc', '-fopenmp', './task2/sieve.c', '-o', './task2/a.out']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p.wait()
    if not os.path.isfile("./task3/gauss-parallel"):
        cmd = ['gcc', '-fopenmp', './task3/gauss-parallel.c', '-o', './task3/gauss-parallel']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p.wait()
    if not os.path.isfile("./task3/sieve-parallel"):
        cmd = ['gcc', '-fopenmp', '-lm', './task3/sieve-parallel.c', '-o', './task3/sieve-parallel']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p.wait()

    print "\nData generator for OpenMP tests.\n"

    #range for Gauss
    gauss_min = 100
    gauss_max = 800
    di1 = 100

    #range for sieve
    sieve_min = 100000
    sieve_max = 8000000 #100000000
    di2 = 100000

    #number cores
    cores = [2, 4, 8]

    for nr_cores in cores:

        #lists with results
        gauss_serial = list()
        sieve_serial = list()
        gauss_par = list()
        sieve_par = list()

        #Change environment variable for runtime setting of
        #number of cores
        #cmd = ['export', 'OMP_NUM_THREADS='+str(nr_cores)]
        #p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        #p.wait()
        #os.environ["OMP_NUM_THREADS"] = str(nr_cores)

        print "Gauss serial ("+str(nr_cores)+" cores):\n"
        #Executing Gauss elimination
        for i in range(gauss_min, gauss_max, di1):
            cmd = ['./task1/a.out', '-n', str(i)]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            for line in p.stdout:
                if line[:4] == "Exec":
                    gauss_serial.append(float(line.split()[2]))
            p.wait()
            print "n = "+str(i)+" done!"

        print "\nGauss parallel ("+str(nr_cores)+" cores):\n"
        #Executing Gauss elmination
        for i in range(gauss_min, gauss_max, di1):
            cmd = ['./task3/gauss-parallel', '-n', str(i), str(nr_cores)]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            for line in p.stdout:
                if line[:4] == "Exec":
                    gauss_par.append(float(line.split()[2]))
            p.wait()
            print "n = "+str(i)+" done!"

        with open("./results/out_gauss_"+str(nr_cores)+".dat", "w") as file_i1:
            #Writing Gauss elimination output to file
            for indx, elem in enumerate(gauss_serial):
                file_i1.write(str((indx+1)*di1)+"\t"+str(gauss_serial[indx])+"\t"+str(gauss_par[indx]))
                file_i1.write("\n")

        print "\nSieve serial ("+str(nr_cores)+" cores):\n"
        #Executing Sieve of Eratosthenes
        for i in range(sieve_min, sieve_max, di2):
            cmd = ['./task2/a.out', '-n', str(i)]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            for line in p.stdout:
                if line[:4] == "Exec":
                    sieve_serial.append(float(line.split()[2]))
            p.wait()
            print "n = "+str(i)+" done!"

        print "\nSieve parallel ("+str(nr_cores)+" cores):\n"
        #Executing SoE
        for i in range(sieve_min, sieve_max, di2):
            cmd = ['./task3/sieve-parallel', '-n', str(i), str(nr_cores)]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            for line in p.stdout:
                if line[:4] == "Exec":
                    sieve_par.append(float(line.split()[2]))
            p.wait()
            print "n = "+str(i)+" done!"

        with open("./results/out_sieve_"+str(nr_cores)+".dat", "w") as file_i2:
            #Writing SoE output to file
            for indx, elem in enumerate(sieve_serial):
                file_i2.write(str((indx+1)*di2)+"\t"+str(sieve_serial[indx])+"\t"+str(sieve_par[indx]))
                file_i2.write("\n")
