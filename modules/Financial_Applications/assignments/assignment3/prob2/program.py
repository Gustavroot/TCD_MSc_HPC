import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from math import sqrt, pow, exp


#Execution:
#	$ python program.py


if __name__ == "__main__":


    nsteps = [4, 8, 16, 32, 64]
    colors = ['b-', 'g-', 'r-', 'y-', 'k-']
    patchs = list()
    patchs.append(mpatches.Patch(color='blue', label='n = 4'))
    patchs.append(mpatches.Patch(color='green', label='n = 8'))
    patchs.append(mpatches.Patch(color='red', label='n = 16'))
    patchs.append(mpatches.Patch(color='yellow', label='n = 32'))
    patchs.append(mpatches.Patch(color='black', label='n = 64'))
    #up to t=2
    y = 2
    
    plt.title("Analyt. Solution to X(t)")

    plt.ylabel("X_(nt)")
    plt.xlabel("nt")
    
    for indx_0, n in enumerate(nsteps):

        #creation of 'x' axis (with re-scaling)
        t = np.arange(1, nsteps[len(nsteps)-1]*y+1, int( nsteps[len(nsteps)-1] / n ))
        steps = np.random.randint(0, 2, size = n*y)
        steps = np.where(steps > 0, 1, -1)
        steps = steps*(1/sqrt(n))
        rwalk = steps.cumsum()
        
        #adjusting output to analytic solution X(t)
        for indx, z in enumerate(rwalk):
            #in following line, t[indx]/n represents each time
            rwalk[indx] = exp(t[indx]/n+rwalk[indx])

        plt.plot(t, rwalk, colors[indx_0])

    plt.legend(loc=4)
    plt.grid(True)
    
    plt.legend(handles = patchs)

    plt.savefig("./analyt_solution_"+str(y)+".png", dpi=None, facecolor='w', edgecolor='w',
		orientation='portrait', papertype=None, format=None,
		transparent=False)

    plt.clf()
