import matplotlib.pyplot as plt
import ast


file_i = open("out_bu.dat", "r")

x_range = [100, 1000, 10000, 20000, 30000]
thread_sizes = [16, 64, 256, 1024]

y_vals = dict()
for key in thread_sizes:
    y_vals[key] = list()

for i in range(0, 20):
    for key, val in y_vals.items():
        y_vals[key].append(list())

list_line = list()

for line in file_i:
    if line == '':
        continue
    str_buff = line[:-1]
    list_buff = str_buff.replace('[','').split('],')
    if list_buff == ['']:
        continue
    list_buff[2] = list_buff[2][:-2]
    for i, x in enumerate(list_buff):
        list_buff[i] = [float(w.strip()) for w in list_buff[i].split(",")]
    list_line.append(list_buff)

#print list_line

#adding plots to figure
for i, plot_line in enumerate(list_line):
    ctr = 0

    for x in plot_line[1]:
        y_vals[plot_line[0][1]][ctr].append(x)
        ctr += 1

    for x in plot_line[2]:
        y_vals[plot_line[0][1]][ctr].append(x)
        ctr += 1
        
#print y_vals



str_labels = ["norms serial - max", "norms par. - max", "norms serial - Frobenius", "norms par. - Frobenius",
	"norms serial - one", "norms par. - one", "norms serial - inf.", "norms par. - inf",
	"exec. time serial - max", "exec. time par. cut - max", "exec. time par. full - max",
	"exec. time serial - Frobenius", "exec. time par. cut - Frobenius", "exec. time par. full - Frobenius",
	"exec. time serial - one", "exec. time par. cut - one", "exec. time par. full - one",
	"exec. time serial - inf", "exec. time par. cut - inf", "exec. time par. full - inf"]

print "100 - 1000 - 10000 - 20000 - 30000"
print

for key, val in y_vals.items():
    print
    print "*** "+str(key)+" ***"
    print len(val)
    for i in range(0, 20):
        if i < 8 and i%2 == 0:
            print
        elif (i-8)%3 == 0:
            print
        print str_labels[i]
        print val[i]


"""
for key, val in y_vals.items():
    
    #plotting norms
    plt.title("Performance tests")
    plt.xlabel("mat. sizes")
    plt.plot(x_range, val[0], 'b-', label="Target dist.")
    plt.plot(x_range, val[1], 'r-', label="Target dist.")
    plt.savefig("perf_tests_max_norm_"+str(key)+".png", dpi=None, facecolor='w', edgecolor='w',
	orientation='portrait', papertype=None, format=None,
	transparent=False)
    plt.clf()
    
    plt.title("Performance tests")
    plt.xlabel("mat. sizes")
    plt.plot(x_range, val[2], 'b-', label="Target dist.")
    plt.plot(x_range, val[3], 'r-', label="Target dist.")
    plt.savefig("perf_tests_frob_norm_"+str(key)+".png", dpi=None, facecolor='w', edgecolor='w',
	orientation='portrait', papertype=None, format=None,
	transparent=False)
    plt.clf()

    #plotting exec times, net
    #else if ctr >=8 and ctr < 16:
        
    
    #plotting exec times, full
    #else:
        
    plt.clf()
"""
