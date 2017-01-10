#No user input is implemented here, this is only
#for illustration/testing purposes

#The matrix-turning is implemented here by layers:
#	a counter runs from the center of the matrix
#	towards the outer of it, and every layer
#	is rotated at a time

#Function for rotation of matrix by 90deg
def turned(original_matrix):
    #Printing original matrix
    print "\n"
    print "Original matrix:"
    print original_matrix

    #Matrix rotation throught the use of layers, as
    #descripted above; the -1 at the end is because
    #of Python starts counting from 0
    layers_counter = int(len(original_matrix)/2)-1

    #Rotating each layer at a time throught the use of
    #a buffer variable
    while layers_counter >= 0:
        for subcounter in range(layers_counter, len(original_matrix)- \
                layers_counter-1):
            buff_var = original_matrix[layers_counter,subcounter]
            original_matrix[layers_counter,subcounter] = \
                original_matrix[subcounter,len(original_matrix)-1-layers_counter]
            original_matrix[subcounter,len(original_matrix)-1-layers_counter] = \
                original_matrix[len(original_matrix)-1-layers_counter,len(original_matrix)-1-subcounter]
            original_matrix[len(original_matrix)-1-layers_counter, \
                len(original_matrix)-1-subcounter] = original_matrix[len(original_matrix)-1- \
                subcounter,layers_counter]
            original_matrix[len(original_matrix)-1-subcounter,layers_counter] = buff_var
        layers_counter -= 1

    #Printing final output matrix
    print "\n"
    print "Rotated matrix:"
    print original_matrix


#-----MAIN PROGRAM

import numpy

#Matrix definition with numpy
matrix_test = numpy.matrix([[1, 2, 3, 4, 5],[3, 4, 5, 6, 7], \
    [5, 6, 7, 8, 9], [5, 6, 7, 8, 9], [5, 6, 7, 8, 9]])

#Call to the function for matrix rotation
turned(matrix_test)
