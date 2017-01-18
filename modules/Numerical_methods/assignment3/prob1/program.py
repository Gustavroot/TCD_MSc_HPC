#Program to solve linear equations using LU decomposition with partial pivot

#Execution:
#	$ python program.py

import numpy as np


#Function for matrix multiplication
def matrix_mult(A, B):
    TB = zip(*B)
    return [[sum(ea*eb for ea,eb in zip(a,b)) for b in TB] for a in A]


#Function for obtaining the solution vector, once L, U, and P have
#been obtained
def solve_system(L, U, P, b):
    #dimension
    N = len(b)

    #replacing b for Pb
    buffer_b = range(N)
    for index_i, b_i in enumerate(b):
        partial_sum = 0
        for index_j, P_ij in enumerate(P[index_i]):
            partial_sum += b[index_j]*P_ij
        buffer_b[index_i] = partial_sum
    b = buffer_b

    #obtaining the vector y
    y = [0]*N
    for index_i, b_i in enumerate(b):
        y[index_i] += b_i
        for j in range(index_i):
            y[index_i] -= L[index_i][j]*y[j]
        y[index_i] /= L[index_i][index_i]

    #obtaining the vector x
    x = [0]*N
    for index_i, y_i in enumerate(y):
        j = index_i+1
        x[index_i] += y_i
        while j<N:
            x[index_i] -= U[index_i][j]*x[j]
            j += 1
    return x



#Function for printing matrix to the terminal
def print_matrix(matr_to_prnt):
    for x in matr_to_prnt:
        buff_str = ""
        for y in x:
            buff_str += "{0:.2f}".format(y)+"\t"
        print buff_str
 

#Function for obtaining the P matrix for the partial pivote
def pivotize(m):
    """Creates the pivoting matrix for m."""
    n = len(m)
    ID = [[float(i == j) for i in xrange(n)] for j in xrange(n)]
    for j in xrange(n):
        row = max(xrange(j, n), key=lambda i: abs(m[i][j]))
        if j != row:
            ID[j], ID[row] = ID[row], ID[j]
    return ID
 

#The LU decomposition step
def lu_decomp(A):
    N = len(A)
    #creating the space for L, U, as 'zero' matrices
    L = [[0.0]*N for i in xrange(N)]
    U = [[0.0]*N for i in xrange(N)]
    #creating the matrix P representing the changes in A, to pivot
    P = pivotize(A)
    #A2 is the matrix to be decomposed as the product L*U
    A2 = matrix_mult(P, A)
    #application of the algorithm (equations 1.1 to 1.3 in 'assignment_3.pdf')
    for j in xrange(N):
        L[j][j] = 1.0
        for i in xrange(j+1):
            s1 = sum(U[k][j]*L[i][k] for k in xrange(i))
            U[i][j] = A2[i][j] - s1
        for i in xrange(j, N):
            s2 = sum(U[k][j]*L[i][k] for k in xrange(j))
            L[i][j] = (A2[i][j] - s2) / U[j][j]
    #finally, returning the matrices necessary to finalize the solution
    return (L, U, P)



#---------
#MAIN code

print "Program to implement LU decomposition with partial pivot.\n"

#part 1.a
matrix_a = list()
vector_b_part_a = list()
#from file:
file_inp1 = open("matrix_input_a.txt", 'r')
buff_ctr = 0
for line in file_inp1:
    #omitting intermediate lines
    if buff_ctr==0 or buff_ctr==1 or buff_ctr==3:
        buff_ctr += 1
        continue
    #extracting vector b
    if buff_ctr==2:
        for x in line.split():
            vector_b_part_a.append(float(x))
        buff_ctr += 1
        continue
    #extracting A
    matrix_a.append(list())
    for x in line.split():
        matrix_a[len(matrix_a)-1].append(float(x))
    buff_ctr += 1
file_inp1.close()
#and applying lu decomposition to the input of part a
l_a = lu_decomp(matrix_a)
u_a = l_a[1]
p_a = l_a[2]
l_a = l_a[0]
#and printing the solution
print "***Solution part a:***"
print "-- matrix A:"
print_matrix(matrix_a)
print "-- vector b: " + str(vector_b_part_a)
print "-- SOLUTION VECTOR x: "+str(solve_system(l_a, u_a, p_a, vector_b_part_a))
print


#**********************
#IMPORTANT NOTE
#    in the following code, the form of the matrix H5 was changed, because
#    the evaluation of elements of the form 1/(i+j-1) leads to indefinite
#    values for the cases i=0,j=1 and i=1,j=0. For better definition, the
#    coefficients of the matrix were changed to the form 1/(i+j+1)
#**********************


#part 2.a
#initializing matrix A and vector b
matrix_b = list()
N = 5
for i in range(N):
    matrix_b.append(list())
    for j in range(N):
        matrix_b[i].append(1.0/(float(i)+float(j)+1.0))
vector_b_part_b = [5.0, 3.550, 2.81428571428571, 2.34642857142857, 2.01746031746032]
#and applying lu decomposition to the input of part a
l_b = lu_decomp(matrix_b)
u_b = l_b[1]
p_b = l_b[2]
l_b = l_b[0]
#and printing the solution
print "***Solution part b:***"
print "-- matrix A:"
print_matrix(matrix_b)
print "-- vector b: " + str(vector_b_part_b)
print "-- SOLUTION VECTOR x: "+str(solve_system(l_b, u_b, p_b, vector_b_part_b))
print


#part 3.a
#initializing matrix A and vector b
matrix_c = list()
N = 5
for i in range(N):
    matrix_c.append(list())
    for j in range(N):
        if i==j: matrix_c[i].append(1)
        elif i-j==1: matrix_c[i].append(4)
        elif i-j==-1: matrix_c[i].append(-4)
        else: matrix_c[i].append(0)
vector_b_part_c = [-4, -7, -6, -5, 16]
#and applying lu decomposition to the input of part a
l_c = lu_decomp(matrix_c)
u_c = l_c[1]
p_c = l_c[2]
l_c = l_c[0]
#and printing the solution
print "***Solution part c:***"
print "-- matrix A:"
print_matrix(matrix_c)
print "-- vector b: " + str(vector_b_part_c)
print "-- SOLUTION VECTOR x: "+str(solve_system(l_c, u_c, p_c, vector_b_part_c))
print
