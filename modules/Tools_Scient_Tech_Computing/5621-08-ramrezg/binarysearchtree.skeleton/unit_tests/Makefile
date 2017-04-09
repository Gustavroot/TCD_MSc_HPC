
all: t/wvtest

# EDIT HERE: add any "../*.c" files that you need to test
t/wvtest: wvtestmain.c wvtest.c t/wvtest.t.c ../bst.c
	gcc -D WVTEST_CONFIGURED -o $@ -I. $^

runtests: all
	t/wvtest

test: all
	./wvtestrun $(MAKE) runtests

clean::
	rm -f *~ t/*~ *.o t/*.o t/wvtest
