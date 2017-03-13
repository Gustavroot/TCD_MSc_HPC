import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from math import sqrt, pow


#Execution:
#	$ python program.py


if __name__ == "__main__":


    #CODE PART A -- 40 random walk paths
    x = 500
    y = 2
    
    plt.title("Random walk.")

    plt.ylabel("W_(nt)")
    plt.xlabel("nt")
    
    for i in range(0, 40):

        t = np.arange(1, x*y+1)
        steps = np.random.randint(0, 2, size = x*y)
        steps = np.where(steps > 0, 1, -1)
        steps = steps*(1/sqrt(x))
        rwalk = steps.cumsum()
    
        plt.plot(t, rwalk, 'b-')

    plt.legend(loc=4)
    plt.grid(True)

    plt.savefig("./part_A/rnd_walk_"+str(x)+"_"+str(y)+".png", dpi=None, facecolor='w', edgecolor='w',
		orientation='portrait', papertype=None, format=None,
		transparent=False)

    plt.clf()


    #CODE PART B
    x = 1000
    y_vals = [1.0, 2.0]
    
    for y in y_vals:
    
        arr_final_values = list()
     
        for i in range(0, 10000):

            t = np.arange(1, x*y+1)
            steps = np.random.randn(int(x*y))
            steps = np.where(steps > 0, 1, -1)
            steps = steps*(1/sqrt(x))
            rwalk = steps.cumsum()
    
            #saving last value of rand walk
            arr_final_values.append(rwalk[len(rwalk)-1])
        
        #plotting histogram and fitting data
    
        mu, sigma = 0, y
    
        # histogram of the data
        n, bins, patches = plt.hist(arr_final_values, 100, normed=1, facecolor='green', alpha=0.75)

        # add a 'best fit' line
        y_plt = mlab.normpdf( bins, mu, sigma)
        l = plt.plot(bins, y_plt, 'r--', linewidth=1)

        plt.xlabel('nt')
        plt.ylabel('W_nt')
        plt.title("Histogram final values.")
        plt.grid(True)

        plt.savefig("./part_B/hist_final_values_t"+str(y)+".png", dpi=None, facecolor='w', edgecolor='w',
		orientation='portrait', papertype=None, format=None,
		transparent=False)
    
        plt.clf() 
