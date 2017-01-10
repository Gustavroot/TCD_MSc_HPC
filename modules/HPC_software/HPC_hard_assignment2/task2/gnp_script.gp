set yrange [0:0.7]
plot "execution_times.dat" using 1:2 with lines

#set yrange [0:0.7]
set term png
set output 'exec_times_curve.png'
replot
