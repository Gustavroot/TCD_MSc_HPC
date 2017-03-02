# Makefile
# $< means first dependency of the current target
# $+ means all current dependencies
# $@ means current target

# Variables for the makefile
CC = mpicc
CFLAGS = -Wall -O2  -ffast-math
LDFLAGS = -lm -ffast-math

MPICC      	= nvcc -Xptxas -v 
#MPI_INCLUDES    = /usr/mpi/intel/mvapich2-1.2p1/include 
MPI_INCLUDES    = /usr/include/openmpi
#MPI_LIBS   	= /usr/mpi/intel/mvapich2-1.2p1/lib 
MPI_LIBS   	= /usr/lib/openmpi


# default target
cudaMpiHelloWorld.exec : cudaMpiHelloWorld.o 
	$(MPICC) -L$(MPI_LIBS) -lmpi -o $@ *.o 


%.o : %.cu 
	$(MPICC) -I$(MPI_INCLUDES) -o $@ -c $< 

#cudaMpiHelloWorld.out: cudaMpiHelloWorld.o
#	echo "Makefile - target cudaMpiHelloWorld.out"
#	$(CC) $(LDFLAGS) -o $@ $+

# Those are not necessary
#%.o: %.c
#	$(CC) -c $< $(CFLAGS)

test: cudaMpiHelloWorld.exe
	./cudaMpiHelloWorld.exe

#clean
.PHONY: clean
clean:
	rm -f *.o *.exe
