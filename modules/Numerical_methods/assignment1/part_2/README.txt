For understanding the theoretical part of the implementations go to sections 2, 3, 4, and for a discussion on the results, go to section 5, of the file ./spline_interp_description.pdf.

This directory contains a program for spline approximation of functions.

If a new function is to be implemented, follow these steps:
 -- go to lines 273 and 283 in ./cubic_spline.py, and chance "runge_function" for whatever the name of the new function is
 -- then, go to the beginning of ./cubic_spline.py, before line 18 (i.e. before the definition of runge_function), and define your own function

If more approximations are to be added, then go to line 259 in ./cubic_spline.py and add more entries to the "grid_points" list

To change the interval, change the associated values at lines 262 and 263 in ./cubic_spline.py.

To execute the Python file and generate the gnuplot files, execute:
	$ python cubic_spline.py

Finally, to generate the plots:
	$ ./call_gnuplot.sh

(if this last command fails, try to add permissions to the file: chmod +x call_gnuplot.sh)

To make the whole process quicker, a bash script to wrap al the previously mentioned steps was written, and can be called like this:
	$ ./general.sh
