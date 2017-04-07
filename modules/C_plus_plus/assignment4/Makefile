CC = g++
CFLAGS = -Wall
LD_FLAGS = -lm

DIR = .

TARG = $(DIR)/agent_based_aquatic

DEP1 = $(DIR)/agent_based_aquatic.o
DEP2 = $(DIR)/external_functions.o
DEP3 = $(DIR)/grid.o
DEP4 = $(DIR)/fish.o

SRC1 = $(DIR)/agent_based_aquatic.cc
SRC2 = $(DIR)/external_functions.cc
SRC3 = $(DIR)/grid.cc
SRC4 = $(DIR)/fish.cc

HEAD2 = $(DIR)/external_functions.h
HEAD3 = $(DIR)/grid.h
HEAD4 = $(DIR)/fish.h

ALL_DEPS = $(DEP1) $(DEP2) $(DEP3) $(DEP4)


all: $(TARG)

#target: agent_based_aquatic
$(TARG): $(ALL_DEPS)
	$(CC) -o $(TARG) $(ALL_DEPS)

#core agent_based_aquatic code
$(DEP1): $(SRC1) $(DEP2) $(DEP3) $(DEP4)
	$(CC) -c $(SRC1) -o $(DEP1)

#external functions
$(DEP2): $(SRC2) $(HEAD2)
	$(CC) -c $(SRC2) -o $(DEP2)

#grid class
$(DEP3): $(SRC3) $(HEAD3)
	$(CC) -c $(SRC3) -o $(DEP3)

#fish class
$(DEP4): $(SRC4) $(HEAD4)
	$(CC) -c $(SRC4) -o $(DEP4)

#random nr generator: ranlux
#ranlux.o: ./ranlux-3.3/ranlxd.c
#	g++ -c ./ranlux-3.3/ranlxd.c -o ./ranlux-3.3/ranlxd.o

#tests
test:
	./$(TARG) #-s 100000 -i 0.1 -f 8 -d 0.05 -D 2 -L 2 -b 100
#	python $(DIR)/plot.py

#removing objects and target
clean:
	rm $(DIR)/*.o #2>/dev/null
	rm $(TARG) #2> /dev/null
