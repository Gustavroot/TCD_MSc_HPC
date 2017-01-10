import numpy

#function for printing matrices
def matrix_print(mat_ex):
    for m_e in mat_ex:
        buff_str = ""
        for s_e in m_e:
            buff_str += str(s_e)+"\t"
        print buff_str+"\n"
    print "\n"

#input file
file_inp = open("input_mat_py.txt", 'r')

#necessary lists and dicts
list_for_A = list()
list_for_B = list()
dict_matrices = dict()
dict_matrices["matrixA"] = list_for_A
dict_matrices["matrixB"] = list_for_B

#getting matrices from input file
buff_str = ""
for line in file_inp:
    if line == "matrixA\n" or line == "matrixB\n":
        buff_str = line[0:-1]
        continue
    dict_matrices[buff_str].append(line.split())
    for index, item in enumerate(dict_matrices[buff_str][-1]):
        dict_matrices[buff_str][-1][index] = float(dict_matrices[buff_str][-1][index])

output_matrix = list()
#matrix multiplication
"""
for index_row_A, row_A in enumerate(dict_matrices["matrixA"]):
    buff_list = list()
    for x in range(0, len(dict_matrices["matrixA"])):
        partial_sum = 0
        for index_row_A_x, k in enumerate(row_A):
            partial_sum += k*dict_matrices["matrixB"][index_row_A_x][x]
        buff_list.append(partial_sum)
    output_matrix.append(buff_list)
"""
matr_A = numpy.matrix(dict_matrices["matrixA"])
matr_B = numpy.matrix(dict_matrices["matrixB"])
output_matrix = numpy.dot(matr_A, matr_B)

#print matrices
matrix_print(dict_matrices["matrixA"])
matrix_print(dict_matrices["matrixB"])
matrix_print(output_matrix.tolist())
