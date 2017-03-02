# Compilers and commands
CC=		g++
CXX=		g++
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
SOURCES		= thread3d.cu
OBJECTS		= thread3d.o

TARGET 		= thread3d


all: main

main: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $(TARGET) -I$(INCPATH)

thread3d.o: thread3d.cu
	$(NVCC)  thread3d.cu -c $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)
