#Execution instructions in the terminal:
#	$ python markov_chain_sim.py

#As the matrix multiplication is small, can
#be put here explicitly
def matrix_mult(x, y, s_c, alpha_c, beta_c, gamma_c, delta_c):
    common_den = s_c+alpha_c*y+beta_c*x*y+gamma_c*x*y+delta_c*x
    output_list = list()
    output_list.append(x+y*((gamma_c*x-delta_c*x/y)/common_den))
    output_list.append(x*((alpha_c*y/x-beta_c*y)/common_den)+y)
    return output_list

#Constants:
s_ct = 0.1
alpha_ct = 0.1
beta_ct = 0.1
gamma_ct = 0.1
delta_ct = 0.1


#Initial values:
x_val = 100.0
y_val = 1000.0


gen_counter = 0
#Evolve 100 times
for k in range(0, 10000):
    commt_markov = matrix_mult(x_val, y_val, s_ct, alpha_ct, beta_ct, gamma_ct, delta_ct)
    print commt_markov
    x_val = commt_markov[0]
    y_val = commt_markov[1]
    if x_val < 1:
        print "\nAll predators died! Remaining preys: "+str(int(y_val))
        break
    elif y_val < 1:
        print "\nAll preys died! Remaining predators: "+str(int(x_val))
        break
    gen_counter += 1

print "\nAmount of total iterations: "+str(gen_counter)+"\n"
