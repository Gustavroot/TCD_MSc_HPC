#Execution instructions:
#	$ python rc_credit.py
#Using Python 2.7


#Imports
from rc_credit_utils import *



#Main code
if __name__ == '__main__':

    #parameters of the simulation
    alpha = 1.4
    pd = 0.01

    print "\nProgram to calculate RC credit from ORE output.\n\n"

    print rc_credit(alpha, pd)
