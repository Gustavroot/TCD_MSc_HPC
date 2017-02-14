CC = g++ -std=c++11
CFLAGS = -Wall

#outp = matrix

#default target
all: testMatrix.cc Matrix.o
	$(CC) -o testMatrix testMatrix.cc Matrix.o

Matrix.o: Matrix.cc Matrix.h
	$(CC) -c Matrix.cc

#cleaning compiled files
clean:
	rm *.o
