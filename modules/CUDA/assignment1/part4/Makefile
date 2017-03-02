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
SOURCES= norms.cu 
OBJECTS= norms.o \
		  mat_norms_parallel.o  

TARGET= mat_norms_parallel

all: mat_norms_parallel

mat_norms_parallel: mat_norms_parallel.o norms.o 
	$(NVCC) norms.o  mat_norms_parallel.o -o mat_norms_parallel -I$(INCPATH)

norms.o: norms.cu 
	$(NVCC)  norms.cu -c $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)
