import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from math import sqrt, pow


#Execution:
#	$ python program.py


#Core functions

#Plotting data
def plot_f(out_name, title, xlabel, ylabel, x_values_array, y_values_array, write_over_plt):

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
    if(not write_over_plt):
        plt.clf()



if __name__ == "__main__":


    #CODE PART B

    nsteps = [100, 200, 1000]
    t_intervals = [0.5, 1.0, 2.0]

    for x in nsteps:
        for y in t_intervals:
            
            #dt = t_intervals[0]/float(npsteps[0])

            t = np.arange(1, x*y+1)
            steps = np.random.randint(0, 2, size = x*y)
            steps = np.where(steps > 0, 1, -1)
            steps = steps*(1/sqrt(x))
            rwalk = steps.cumsum()
    
            #stored as "rnd_walk_nsteps_t.png"
            plot_f("./part_B/rnd_walk_"+str(x)+"_"+str(y)+".png", "Random walk.", "nt", "W_(nt)", t, rwalk, 0)


    #CODE PART C
    x = 500
    y = 2
    
    plt.title("Random walk.")

    plt.ylabel("W_(nt)")
    plt.xlabel("nt")
    
    for i in range(0, 400):

        t = np.arange(1, x*y+1)
        steps = np.random.randint(0, 2, size = x*y)
        steps = np.where(steps > 0, 1, -1)
        steps = steps*(1/sqrt(x))
        rwalk = steps.cumsum()
    
        plt.plot(t, rwalk, 'b-')

    plt.legend(loc=4)
    plt.grid(True)

    plt.savefig("./part_C/rnd_walk_"+str(x)+"_"+str(y)+".png", dpi=None, facecolor='w', edgecolor='w',
		orientation='portrait', papertype=None, format=None,
		transparent=False)

    plt.clf()


    #CODE PART D
    x = 1000
    y_vals = [0.5, 2.0]
    
    for y in y_vals:
    
        arr_final_values = list()
     
        for i in range(0, 10000):

            t = np.arange(1, x*y+1)
            steps = np.random.randint(0, 2, size = x*y)
            steps = np.where(steps > 0, 1, -1)
            steps = steps*(1/sqrt(x))
            rwalk = steps.cumsum()
    
            #saving last value of rand walk
            arr_final_values.append(rwalk[len(rwalk)-1])
        
        #plotting histogram and fitting data
    
        mu, sigma = 0, y
    
        # histogram of the data
        n, bins, patches = plt.hist(arr_final_values, 50, normed=1, facecolor='green', alpha=0.75)

        # add a 'best fit' line
        y_plt = mlab.normpdf( bins, mu, sigma)
        l = plt.plot(bins, y_plt, 'r--', linewidth=1)

        plt.xlabel('nt')
        plt.ylabel('W_nt')
        plt.title("Histogram final values.")
        plt.grid(True)

        plt.savefig("./part_D/hist_final_values_t"+str(y)+".png", dpi=None, facecolor='w', edgecolor='w',
		orientation='portrait', papertype=None, format=None,
		transparent=False)
    
        plt.clf() 
