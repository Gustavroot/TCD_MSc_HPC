import struct

#Program to convert numbers into IEEE 754 single precision numbers

#TODO: read https://docs.python.org/2/tutorial/floatingpoint.html

#Program execution:
#	python single_prec_conversion.py

#To compare at the end of the conversion, the following
#function uses 'struct' function to obtain the desired representation
#(taken from: http://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex)
def direct_conversion(num):
    return ''.join(bin(ord(c)).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))


#Function for conversion
def convert_to_ieee_754(orig_number):
    final_ieee_nr = ""
    #Sign digit
    sign_digit = ""
    if orig_number > 0:
        sign_digit += "0"
    else:
        sign_digit += "1"
        orig_number = float(str(orig_number)[1:])
    #..alternative optimized choice
    nr_buff = float(orig_number)
    nr_buff_str = str(nr_buff)
    nr_buff_parts = nr_buff_str.split(".")
    power_p_bin = bin(len(bin(int(nr_buff_parts[0]))[2:])-1+127)[2:]
    f_first_part = bin(int(nr_buff_parts[0]))[3:]
    f_second_part = ""
    if int(nr_buff_parts[1]) != 0:
        f_second_part += bin(int(nr_buff_parts[1]))[2:]
    f_nr = f_first_part+f_second_part
    #Filling zeros in f
    for k in range(len(f_nr), 23):
        f_nr += "0"
    final_ieee_nr += sign_digit + power_p_bin + f_nr
    return final_ieee_nr


#Function for conversion
def binary_to_hexadecimal(orig_binary_number):
    conversion_dict = {"0000": "0", "0001": "1", "0010": "2", "0011": "3", "0100": "4", "0101": "5", "0110": "6", "0111": "7", "1000": "8", "1001": "9", "1010": "A", "1011": "B", "1100": "C", "1101": "D", "1110": "E", "1111": "F"}

    final_hexa_nr = ""
    for k in range(0, len(orig_binary_number)-4, 4):
        final_hexa_nr += conversion_dict[orig_binary_number[k:k+4]]
    return final_hexa_nr


#---MAIN PROGRAM
print "\nProgram for converting a number into IEEE 754 single prec.\n"

#------------------first case: 55
ieee_nr = convert_to_ieee_754(55)
hexad_nr = binary_to_hexadecimal(ieee_nr)
print "\n55 ------\nWith the use of the implemented code: "
print "**single prec. binary: "+str(ieee_nr)
print "**hexadecimal: "+str(hexad_nr)
print "With the use of Python function 'struct':"
print "**single prec. binary: "+direct_conversion(55)
#print "hexadecimal: "+direct_conversion(55)
#--------------------------------

#------------------second case: 55.5
ieee_nr = convert_to_ieee_754(55.5)
hexad_nr = binary_to_hexadecimal(ieee_nr)
print "\n55.5 ------\nWith the use of the implemented code: "
print "**single prec. binary: "+ieee_nr
print "**hexadecimal: "+binary_to_hexadecimal(ieee_nr)
print "**hexad check (python-like implementation): "+hex(int(ieee_nr, 2))
print "With the use of Python function 'struct':"
print "**single prec. binary: "+direct_conversion(55.5)
#print "hexadecimal: "+direct_conversion(55.5)
print ""
#-----------------------------------
