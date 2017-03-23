import sys

#TODO: set HPL running values from a simple .csv file

#TODO: automatically set the PxQ combinations from
#      the multiplicants constituting sys.argv[1]

#main code
if __name__ == "__main__":

    #open HPL.dat
    file_in = open("bin/"+sys.argv[2]+"/HPL_buff.dat", "r")
    file_out = open("bin/"+sys.argv[2]+"/HPL.dat", "w")

    nr_cores = sys.argv[1]

    #calculate dynamically P and Q from nr_cores

    for line in file_in:
        if len(line.split("#"))>1 and line.split("#")[1] == " of problems sizes (N)\n":
            string_buff = "#".join(["1            ", line.split("#")[1]])
        elif line.split()[len(line.split())-1] == "Ns":
            string_buff = str(float(nr_cores)*28000)+"        Ns\n"
        elif len(line.split("#"))>1 and line.split("#")[1] == " of NBs\n":
            string_buff = "#".join(["4            ", line.split("#")[1]])
        elif line.split()[len(line.split())-1] == "NBs":
            string_buff = "90 140 180 220 NBs\n"
        elif len(line.split("#"))>1 and line.split("#")[1] == " of process grids (P x Q)\n":
            string_buff = "#".join(["1            ", line.split("#")[1]])
        elif line.split()[len(line.split())-1] == "Ps":
            string_buff = "2            Ps\n"
        elif line.split()[len(line.split())-1] == "Qs":
            string_buff = "4            Qs\n"
        else:
            string_buff = line

        file_out.write(string_buff)

    file_in.close()
    file_out.close()
