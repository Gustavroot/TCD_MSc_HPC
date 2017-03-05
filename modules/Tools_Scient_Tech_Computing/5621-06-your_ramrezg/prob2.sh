#!/bin/bash

#Execution instructions:
#	$ ./prob2.sh 1 2 -3


#Checking number of input params
if [ $# -ne 3 ]
then
    echo "Wrong number of params!"
    exit
fi

#Checking that input params are int

#Sources:
#http://grzechu.blogspot.de/2006/06/bash-scripting-
#	checking-if-variable-is.html
#http://stackoverflow.com/questions/13790763/bash-
#	regex-to-check-floating-point-numbers-from-user-input

for num in $*
    do
    #Check if input is valid number:
    if ! [[ $num =~ ^[+-]?[0-9]+\.?[0-9]*$ ]]
    then
        exit
    fi

    #Alternative method:
    #if [ $num -eq $num 2> /dev/null ]
    #then
    #    echo "$num is a number"
    #else
    #    echo "$num isn't a number"
    #    exit
    #fi
done

A=$1
B=$2
C=$3

#'A' has to be diff to zero
if [ $A -eq 0 ]
then
    exit
fi

echo
echo "Input info:"
echo -e "\t A=$A"
echo -e "\t B=$B"
echo -e "\t C=$C"

echo -e "\nsolving system...\n"

#check for discriminant value
disc=$(echo "$B * $B - 4 * $A * $C" | bc -l)
bool_test=$(echo "$disc < 0" | bc -l)
if [ $bool_test -eq 1 ]
then
    echo "system has negative determinant!"
    exit
fi

#solve the system

result1=$(echo "scale=4; ( - $B + sqrt ( $disc ) ) / ( 2 * $A )" | bc -l)
result2=$(echo "scale=4; ( - $B - sqrt ( $disc ) ) / ( 2 * $A )" | bc -l)

#display results
echo "Solutions:"

echo -e "\tresult = $result1"
echo -e "\tresult = $result2"

echo
