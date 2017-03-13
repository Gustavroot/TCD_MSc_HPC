#Script to plot data files for OpenMP tests

import subprocess

import matplotlib.pyplot as plt


if __name__ == "__main__":

    print "\nPlotting data from OpenMP tests.\n"

    input_files = list()

    #open input files
    cmd = ['ls', './results/']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    for line in p.stdout:
        if line[:3] == "out":
            input_files.append(line[:-1])
    p.wait()

    for file_i in input_files:
        with open("./results/"+file_i, "r") as file_var:

            #in for [x, y]
            serial = [list(), list()]
            parallel = [list(), list()]

            for line in file_var:
                buff = line[:-1]
                serial[0].append(float(buff.split()[0]))
                parallel[0].append(float(buff.split()[0]))
                serial[1].append(float(buff.split()[1]))
                parallel[1].append(float(buff.split()[2]))

            #re-scaling x axis for Sieve of Er.
            if file_i.split("_")[1] == "sieve":
                for indx, w in enumerate(serial[0]):
                    serial[0][indx] = serial[0][indx]/1000000
                    parallel[0][indx] = parallel[0][indx]/1000000

            plt.title(file_i.split("_")[1]+" with "+file_i.split("_")[2].split(".")[0]+" cores")
            #print file_i.split("_")[1]+" with "+file_i.split("_")[2].split(".")[0]+" cores"
            plt.ylabel("exec. time")
            if file_i.split("_")[1] == "gauss":
                plt.xlabel("nr of variables")
            else:
                plt.xlabel("max N ( x 10^6 )")

            plt.plot(serial[0], serial[1], 'b-', label="serial")
            plt.plot(parallel[0], parallel[1], 'r-', label="parallel")
            plt.legend(loc=4)
            plt.grid(True)
            plt.savefig("./plots/"+file_i.split("_")[1]+"_"+file_i.split("_")[2].split(".")[0]+".png",
            	dpi=None, facecolor='w', edgecolor='w',
		orientation='portrait', papertype=None, format=None,
		transparent=False)
            plt.clf()
