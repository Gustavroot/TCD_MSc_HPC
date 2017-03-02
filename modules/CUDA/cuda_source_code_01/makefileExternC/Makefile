# Compilers and commands
CC=		gcc
CXX=		gcc
NVCC=		nvcc
LINK=		nvcc
DEL_FILE= 	rm -f

#Flags
#PARALLEL	= -fopenmp
#DEFINES		= -DWITH_OPENMP
CFLAGS		= -W -Wall $(PARALLEL) $(DEFINES)
CXXFLAGS	= -W -Wall $(PARALLEL) $(DEFINES)
NVCCFLAGS	= -g -G --use_fast_math

LIBS		= $(PARALLEL)
INCPATH		= /usr/include/
# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops


####### Files
SOURCES		= add_vectors.cu
OBJECTS		= add_vectors.o \
		  main.o

TARGET 		= add_vectors


all: main

main: main.o add_vectors.o
#	$(NVCC) main.c add_vectors.cu -c $(NVCCFLAGS) -I$(INCPATH)
	$(NVCC) add_vectors.o  main.o -o $(TARGET) -I$(INCPATH)

add_vectors.o: add_vectors.cu
	$(NVCC)  add_vectors.cu -c $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)
