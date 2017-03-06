CC=g++


#all: connect_4

connect_4: connect_4.o connect_4.cc
	$(CC) -o connect_4 connect_4.o user.o board.o

connect_4.o: user.o board.o connect_4.cc
	$(CC) -c connect_4.cc

user.o: user.cc user.h
	$(CC) -c user.cc

board.o: board.cc board.h
	$(CC) -c board.cc

test:
	./connect_4

clean:
	rm *.o
	rm connect_4
