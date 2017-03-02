# Makefile

# Compilers and commands
CC=		g++
CXX=		g++
NVCC=		nvcc
LINK=		nvcc


#Flags
PARALLEL	= -fopenmp
NVCC_PARALLEL	= -Xcompiler -fopenmp
# NVCC_PARALLEL	= -Xcompiler -openmp
#DEFINES		= -DWITH_OPENMP
CFLAGS		= -W -Wall -O4 $(PARALLEL) $(DEFINES)
CXXFLAGS	= -W -Wall -O4 $(PARALLEL) $(DEFINES)
NVCCFLAGS	= -O4 --use_fast_math -lm $(NVCC_PARALLEL)

LIBS		= $(PARALLEL) -lm
INCPATH		= /usr/include/

# Old versions
#CFLAGS = -O4 -fopenmp
#CFLAGS = -g -fopenmp
#LDFLAGS = -lm
DEL_FILE= 	rm -f


####### Files
OBJECTS		= cudaOpenMPHelloWorld.o

TARGET 		= cudaOpenMPHelloWorld.out

# default target
all: main

main: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $(TARGET) -I$(INCPATH) $(NVCCFLAGS)

%.o : %.cu 
	$(NVCC) -o $@ -c $<  $(NVCCFLAGS) -I$(INCPATH)

#clean
.PHONY: clean
clean:
	rm -f *.o *.exe
