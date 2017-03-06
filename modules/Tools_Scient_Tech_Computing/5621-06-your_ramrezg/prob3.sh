#!/bin/bash


#ARRAY=()
general_ctr=0

while [ $general_ctr -lt 7 ]
do
    #create random number
    rnd=$(( ( RANDOM % 45 )  + 1 ))
    echo "new rnd nr: $rnd"
    
    if [ -z "$ARRAY" ]
    then
        echo "empty!"
        ARRAY[0]="$rnd"
        echo "after new insertion: $ARRAY"
        ((general_ctr++))
        echo $general_ctr
    else
        #check that number is not equal to previous ones
        for prev in "${ARRAY[@]}"
        do
            if [ $rnd -eq $prev ]
            then
                echo "equal!"
            else
                echo $ARRAY
                echo "not equal!"
                #increase counter and append
                ARRAY[$general_ctr]="$rnd"
                echo "after new insertion: $ARRAY"
                ((general_ctr++))
                echo $general_ctr
            fi
        done
    fi

done
