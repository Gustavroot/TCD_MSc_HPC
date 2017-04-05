# Compilers and commands
CC=		gcc
CXX=		gcc
NVCC=		nvcc

LINK=		nvcc
DEL_FILE= 	rm -f

#Flags
#PARALLEL	= -fopenmp
#DEFINES= -DWITH_OPENMP

CFLAGS		= -W -Wall $(PARALLEL) $(DEFINES)
CXXFLAGS	= -W -Wall $(PARALLEL) $(DEFINES)
NVCCFLAGS	= -g -G --use_fast_math
LIBS		= $(PARALLEL)
INCPATH		= /usr/include/

# Old versions
#CFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
#CXXFLAGS=-lGL -lglut -lpthread -llibtiff  -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops

####### Files
SOURCES= sys_evol_gpu.cu
OBJECTS= sys_evol_gpu.o \
		  cyl_rad.o

TARGET= cyl_rad

all: cyl_rad

cyl_rad: cyl_rad.o sys_evol_gpu.o
	$(NVCC) sys_evol_gpu.o  cyl_rad.o -o cyl_rad -I$(INCPATH)

sys_evol_gpu.o: sys_evol_gpu.cu
	$(NVCC) sys_evol_gpu.cu -c $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)
