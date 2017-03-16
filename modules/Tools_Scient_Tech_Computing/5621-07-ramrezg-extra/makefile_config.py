import sys

#CORE functions



#main code
if __name__ == "__main__":

    #input params
    ARCH = sys.argv[1]
    TOPdir = sys.argv[2]
    MPdir=sys.argv[3]
    
    #supress the tail of the MPdir string
    MPdir = MPdir.split("/")
    MPdir.pop()
    MPdir.pop()
    MPdir = '/'.join(MPdir)
    
    MPlib = sys.argv[4]
    
    LAdir = sys.argv[5]
    LAlib = sys.argv[6]
    
    filename = sys.argv[7]
    
    blas_type = sys.argv[8]
    
    #open Makefiles
    in_file = open("Make.lonsdale_gcc_mkl", "r")
    out_file = open(filename, "w")

    list_lines_out = list()
    #TODO: modify Makefile accordingly
    
    #steps:
    #	 set ARCH
    #	 set TOPdir
    #	 set MPdir
    #	 set MPlib (by changing it's name accordingly)
    #	 set LAdir
    #	 set LAlib
    #	 set CC
    #	 set LINKER

    #for line in in_file:
    #    

    #close Makefiles
    in_file.close()
    out_file.close()
