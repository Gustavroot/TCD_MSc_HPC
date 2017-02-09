CC = g++ -std=c++11
CFLAGS = -Wall
#LDFLAGS = -lm

outp = matrix

#compiling the implementation
$(outp): Matrix.o
	$(CC) -o matrix Matrix.o

Matrix.o: Matrix.h
	$(CC) -c Matrix.cc


#testing the implementation
test: testMatrix
	./testMatrix

testMatrix:
	$(CC) -o testMatrix testMatrix.cc


#cleaning compiled files
clean:
	rm $(outp)
	rm *.o
