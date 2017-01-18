#Implementation of Jacobi and Gauss-Seidel iterative solvers

#TODO: optimize this code (store the 3 diagonals only)

import numpy as np


#Function implementing Jacobi
def jacobi(A, b, N=25, x=None):
    #Create an initial guess if needed.. if no guess passed, then
    #the initial guess is zeros
    if x is None:
        x = np.zeros(len(A[0]))

    #Creation of the diagonal and non-diagonal matrices
    D = np.diag(A)
    R = A - np.diagflat(D)

    #Iterate for N times using the iterative formula
    for i in range(N):
        x = (b - np.dot(R,x)) / D
    return x


#Function implementing Gauss-Seidel
def gauss_seidel(A, b, x, n):
    #extracting the L matrix using numpy
    L = np.tril(A)
    #and then obtaining U
    U = A-L
    for i in range(n):
        x = np.dot(np.linalg.inv(L), b - np.dot(U, x))
        #print str(i).zfill(3),
        #print(x)
    return x


#MAIN code
print "\nProgram for solving a system using both Jacobi and Gauss-Seidel methods.\n"

#read input matrix from file:
input_mtrx = list()
file_inp = open("input_mtrx.txt", 'r')
for line in file_inp:
    input_mtrx.append(list())
    for nr in line.split():
        input_mtrx[len(input_mtrx)-1].append(float(nr))
file_inp.close()

#system data:
A = np.array(input_mtrx)
b = np.array([5, 11, 18, 21, 29, 40, 48, 48, 57, 72, 80, 76, 69, 87, 94, 85])

#number of iterations:
N = 100

#applying Jacob iterative solver
print "Jacobi method:"
guess_v = np.array([1]*len(input_mtrx))
print "-- solution vector: "+str(jacobi(A, b, N, x=guess_v))
print

#applying Gauss-Seidel iterative solver
print "Gauss-Seidel method:"
guess_v = np.array([1]*len(input_mtrx))
print "-- solution vector: "+str(gauss_seidel(A, b, guess_v, N))
print
