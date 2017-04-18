set key outside

set term png
set output 'performances.png'

set logscale y 2

set xlabel "(N-1000)/500 with N = number of tests"
set ylabel "log_2(t_exec)"

plot 'output.dat' using 0:3 with lines title columnheader, '' using 0:5 with lines title columnheader
