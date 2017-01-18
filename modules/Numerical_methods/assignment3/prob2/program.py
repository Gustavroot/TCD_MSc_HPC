#Program implementing the condition number estimator

#Data of k* vs n is saved in two output files:
#	out1.txt for A_n
#	out2.txt for H_n

import numpy as np
from numpy.linalg import inv

#Function returning the value of the condition number for matrix A
def condit_number(A):
    return 0


#Function to create matrix A_n
def matrix_a_n(n):
    buff_matrx = list()
    for i in range(n):
        buff_matrx.append(list())
        for j in range(n):
            if i==j: buff_matrx[i].append(1.0)
            elif i-j==1: buff_matrx[i].append(4.0)
            elif i-j==-1: buff_matrx[i].append(-4.0)
            else: buff_matrx[i].append(0.0)
    return buff_matrx

#Function to create matrix H_n
def matrix_h_n(n):
    buff_matrx = list()
    for i in range(n):
        buff_matrx.append(list())
        for j in range(n):
            buff_matrx[i].append(1.0/(float(i)+float(j)+1))

    return buff_matrx


#Function that calculates the inf norm of a vector
def inf_norm_vectr(v):
    buff_sum = 0
    for x in v:
        if abs(x)>buff_sum: buff_sum = abs(x)
    return buff_sum


#Function that calculates the inf norm of matrix
def inf_norm_matrix(M):
    buff_sum = 0
    for i in range(len(M)):
        buff_sub_sum = 0
        for j in range(len(M[i])):
            buff_sub_sum += abs(M[i][j])
        if buff_sub_sum>buff_sum: buff_sum = buff_sub_sum
    return buff_sum


#Function that implements the algorithm to estimate the condition nr
def condition_nr(A):
    #first, obtaining coefficient alpha
    alpha_nr = inf_norm_matrix(A)
    #taking random initial guess of y_0
    y = [1]*len(A) #list()
    #obtaining the inverse of A
    A_mtrx = np.matrix(A)
    A_inv = inv(A_mtrx)
    A_inv = np.array(A_mtrx)
    y = np.array(y)
    #compute y_5
    for i in range(1, 6, 1):
        y = A_inv.dot(y)*(1.0/inf_norm_vectr(y))
    #inf norm of y_5
    v = inf_norm_vectr(y)
    return alpha_nr*v



#---------
#MAIN code
print "\nEstimating condition number:\n"

file1 = open("out1.txt", 'w')
#matrices H_n
print "matrices H_n:"
for i in range(4, 21, 1):
    file1.write(str(i)+","+str(condition_nr(matrix_h_n(i)))+"\n")
    print str(i)+": "+str(condition_nr(matrix_h_n(i)))
print
file1.close()

file2 = open("out2.txt", 'w')
#matrices A_n
print "matrices A_n:"
for i in range(4, 21, 1):
    file2.write(str(i)+","+str(condition_nr(matrix_a_n(i)))+"\n")
    print str(i)+": "+str(condition_nr(matrix_a_n(i)))
print
file2.close()
