#Approximation plots

set yrange [-0.2:0.8]
set title "Plot 1"
set xlabel "Nr. of randoms"
set ylabel "Average"
set grid
plot "./data_for_plots/cuadratic.dat" title "Plot 1"
set term png
set output 'plots/cuadratic.png'
replot

reset
set yrange [-0.2:0.8]
set title "Plot 2"
set xlabel "Nr. of randoms"
set ylabel "Average"
set grid
plot "./data_for_plots/linear.dat" title "Plot 2"
set term png
set output 'plots/linear.png'
replot

reset
set yrange [-0.2:0.8]
set title "Plot 3"
set xlabel "Nr. of randoms"
set ylabel "Average"
set grid
plot "./data_for_plots/sqrt2.dat" title "Plot 3"
set term png
set output 'plots/sqrt2.png'
replot

#Error plots

reset
set yrange [-0.05:0.1]
set title "Plot 4"
set xlabel "Nr. of randoms"
set ylabel "Error"
set grid
plot "./data_for_plots/errorssqrt2.dat" title "Plot 4"
set term png
set output 'plots/errorssqrt2.png'
replot

reset
set yrange [-0.05:0.1]
set title "Plot 5"
set xlabel "Nr. of randoms"
set ylabel "Error"
set grid
plot "./data_for_plots/errorslinear.dat" title "Plot 5"
set term png
set output 'plots/errorslinear.png'
replot

set yrange [-0.05:0.1]
set title "Plot 6"
set xlabel "Nr. of randoms"
set ylabel "Error"
set grid
plot "./data_for_plots/errorscuadratic.dat" title "Plot 6"
set term png
set output 'plots/errorscuadratic.png'
replot

#Variance plots with exact average

reset
set yrange [-0.05:0.1]
set title "Plot 7"
set xlabel "Nr. of randoms"
set ylabel "Variance"
set grid
plot "./data_for_plots/exactvariancessqrt2.dat" title "Plot 7"
set term png
set output 'plots/exactvariancessqrt2.png'
replot

reset
set yrange [-0.05:0.1]
set title "Plot 8"
set xlabel "Nr. of randoms"
set ylabel "Variance"
set grid
plot "./data_for_plots/exactvarianceslinear.dat" title "Plot 8"
set term png
set output 'plots/exactvarianceslinear.png'
replot

set yrange [-0.05:0.1]
set title "Plot 9"
set xlabel "Nr. of randoms"
set ylabel "Variance"
set grid
plot "./data_for_plots/exactvariancescuadratic.dat" title "Plot 9"
set term png
set output 'plots/exactvariancescuadratic.png'
replot

#Variance plots with approx average

reset
set yrange [-0.05:0.1]
set title "Plot 10"
set xlabel "Nr. of randoms"
set ylabel "Variance"
set grid
plot "./data_for_plots/approxvariancessqrt2.dat" title "Plot 10"
set term png
set output 'plots/approxvariancessqrt2.png'
replot

reset
set yrange [-0.05:0.1]
set title "Plot 11"
set xlabel "Nr. of randoms"
set ylabel "Variance"
set grid
plot "./data_for_plots/approxvarianceslinear.dat" title "Plot 11"
set term png
set output 'plots/approxvarianceslinear.png'
replot

set yrange [-0.05:0.1]
set title "Plot 12"
set xlabel "Nr. of randoms"
set ylabel "Variance"
set grid
plot "./data_for_plots/approxvariancescuadratic.dat" title "Plot 12"
set term png
set output 'plots/approxvariancescuadratic.png'
replot


