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

    C_comp = sys.argv[9]

    #open Makefiles
    in_file = open("setup/Make.Linux_PII_CBLAS", "r")
    out_file = open(filename, "w")

    #steps:
    #	 set ARCH
    #	 set TOPdir
    #	 set MPdir
    #	 set MPlib (by changing it's name accordingly)
    #	 set LAdir
    #	 set LAlib
    #	 set CC
    #	 set LINKER

    for line in in_file:
        if line[:5] == "ARCH ":
            str_buff = "=".join([line.split("=")[0], " "+ARCH+"\n"])
        elif line[:6] == "TOPdir":
            str_buff = "=".join([line.split("=")[0], " "+TOPdir+"\n"])
        elif line[:5] == "MPdir":
            str_buff = "=".join([line.split("=")[0], " "+MPdir+"\n"])
        elif line[:5] == "MPlib":
            str_buff = "/".join(line.split("/")[:-1]+[MPlib+"\n"])
        elif line[:5] == "LAdir":
            str_buff = "=".join([line.split("=")[0], " "+LAdir+"\n"])
        elif line[:5] == "LAlib":
            str_buff = "=".join([line.split("=")[0], " "+LAlib+"\n"])
        elif line[:3] == "CC ":
            str_buff = "=".join([line.split("=")[0], " "+C_comp+"\n"])
        elif line[:6] == "LINKER":
            str_buff = "=".join([line.split("=")[0], " "+C_comp+"\n"])
        elif line[:8] == "HPL_OPTS":
            if blas_type == "acml":
                str_buff = "#"+line
        else:
            str_buff = line
        out_file.write(str_buff)


    #close Makefiles
    in_file.close()
    out_file.close()
