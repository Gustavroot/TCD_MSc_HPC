# set variables
CC=gcc
#CFLAGS=-Wall -O2
CFLAGS=-Wall -g			# turn on debugging - for gdb
#CFLAGS=-Wall -g -DDEBUG	# turn on debugging - in the code
#CFLAGS=-Wall
LDFLAGS=

# custom variables
target         = bst_test
objects        = bst_test.o bst.o

default: $(target)

$(target): $(objects)
	$(CC) $(CFLAGS) $(LDFLAGS) $(objects) -o $@

# explicit dependencies required for headers
bst_test.o:      bst.h
bst.o:           bst.h

# phony target to get around problem of having a file called 'clean'
.PHONY: clean
clean:
	$(RM) $(objects) $(target)
	@ #extra elements created from 'make graph'
	$(RM) output.dat

test: $(target)
	./$(target) -n 1000

unit_tests: $(target).o
	make -C unit_tests test

graph: $(target)
	@make
	@echo "step n_i random n_i linear" > output.dat
	@echo "Performing tests..."
	@for i in $$(seq 1000 500 20000) ; do \
		./$(target) -n $$i; \
	done >> output.dat
	@gnuplot plot_performances.gp
	@echo "...done."
	@make clean
