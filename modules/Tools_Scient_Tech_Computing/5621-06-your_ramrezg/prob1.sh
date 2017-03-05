#!/bin/bash

#1st layer loop: running from 40000 to 42000
for num in $(seq 40000 1 42000)
do
    prime_ctr=0
    dividr=2
    #running from 2 to $num, to check if $num is prime
    while [ $dividr -lt $num ]
    do
        ((mod_buff = $num % $dividr))
        #if $num is divisible by a number less than $num
        if [ $mod_buff -eq 0 ]
        then
            ((prime_ctr++))
            break
        fi
        ((dividr++))
    done
    #in case $num was not divisible by a number less than $num
    if [ $prime_ctr -eq 0 ]
    then
        echo "$num"
    fi
done
