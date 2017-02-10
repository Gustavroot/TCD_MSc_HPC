CC = g++ -std=c++11
CFLAGS = -Wall

#outp = matrix

#compiling the implementation
all: Matrix.o testMatrix
	@echo
	./testMatrix
	@echo

testMatrix: testMatrix.cc Matrix.o
	$(CC) -o testMatrix testMatrix.cc Matrix.o

Matrix.o: Matrix.cc Matrix.h
	$(CC) -c Matrix.cc

#cleaning compiled files
clean:
	rm *.o
