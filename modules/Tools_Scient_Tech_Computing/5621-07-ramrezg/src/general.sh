#!/bin/bash


#TODO: automatically set values for compilers
#      and BLAS libraries from values in 'env'


#Execution instructions
#	$ ./general.sh NR_CORES

#check nr of input params
if [ "$#" != "1" ]
then
    echo "Wrong number of input params."
    exit
fi

#GEN_DIR_NAME="5621-07-ramrezg"

#unload all previous modules
module purge

#set number of cores by command line
nr_cores=$1

#list of compilers
compilers=("gcc/4.9.3-gnu" "intel/cc/64/11.0.074" "intel/12.1/composer_xe_2011_sp1.9.293" "intel/15.0.6/composer_xe_2015.6.233")
compilers_alias=("gcc" "icc" "icc" "icc")
#..following labels are used when naming makefiles
compilers_labels=("gcc" "intel11" "intel12" "intel15")

#parallel compiler support
par_comp_support=("default-gcc-openmpi-4.9.3-1.8.6" "default-intel-openmpi" "openmpi/1.6.5-intel12.1.3" "intel/15.0.6/impi-5.0.3.049")
par_comp_labels=("openmpi" "openmpi" "openmpi" "impi")

#no need to load MKL for last two
blas_support_mkl=("mkl/64/11.0.074" "mkl/64/11.0.074" " " " ")
blas_support_acml=("acml/64/intel/4.3.0" "acml/64/intel/4.3.0" "acml/64/intel/4.3.0" "acml/64/intel/4.3.0")

mplib_list=("libmpi.so" "libmpi.so" "libmpi.so" "libmpich.a")

#LAdir values for MKL
mkl_ladir_list=("/home/support/apps/intel/cc/64/11.0.074/mkl/lib/em64t" \
	"/home/support/apps/intel/cc/64/11.0.074/mkl/lib/em64t" \
	"/home/support/apps/intel/12.1/composer_xe_2011_sp1.9.293/mkl/lib/intel64" \
	"/home/support/apps/intel/15.0.6/composer_xe_2015.6.233/mkl/lib/intel64")
#library link for MKL11
mkl_lalib_list=("-lmkl_intel_lp64 -lmkl_sequential -lmkl_core" "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core" \
	"-lmkl_intel_lp64 -lmkl_sequential -lmkl_core" "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core")

#LAdir values for ACML
acml_ladir_list=("/home/support/apps/libs/acml/4.3.0/ifort64/lib" "/home/support/apps/libs/acml/4.3.0/ifort64/lib" \
	"/home/support/apps/libs/acml/4.3.0/ifort64/lib" "/home/support/apps/libs/acml/4.3.0/ifort64/lib")
#ACML flags
acml_lalib_list=("-lacml" "-lacml" "-lacml" "-lacml")

#make dir to output Makefiles

mkdir makefiles 2> /dev/null
mkdir makefiles/nr_cores_$nr_cores/ 2> /dev/null

#simple labels to control general flow
blas_types=("mkl" "acml")

#For each compiler, do one compilation and execution for
#both MKL and ACML
indx=0
for comp in ${compilers[@]}
do
    for b_t in ${blas_types[@]}
    do
	comp_label="lonsdale_"${compilers_labels[$indx]}"_"${par_comp_labels[$indx]}"_"$b_t
        out_makef_name="Make."$comp_label

        echo "-------------------------------------"
        echo "Compiling for arch="$comp_label
        echo "-------------------------------------"

        #move to 'root' level of dirs in HPL
        tar -xzf hpl-2.1.tar.gz
        cd hpl-2.1/

        #make good modules availability
        module load apps libs cports
        echo "Loaded apps, libs, cports"

        #loading compiler and openmpi support
        module load ${compilers[$indx]} ${par_comp_support[$indx]}
        echo "Loaded compilers and openmpi/mpi support"

        #load mkl and acml modules
        if [ "$b_t" = "mkl" ]
        then
            module load ${blas_support_mkl[$indx]}
            echo "Loaded blas support"
        elif [ "$b_t" = "acml" ]
        then
            module load ${blas_support_acml[$indx]}
            echo "Loaded blas support"
        fi

        #set LAdir and LAlib variables
        if [ "$b_t" = "mkl" ]
        then
            ladir=${mkl_ladir_list[$indx]}
        elif [ "$b_t" = "acml" ]
        then
            ladir=${acml_ladir_list[$indx]}
        fi

        if [ "$b_t" = "mkl" ]
        then
            lalib=${mkl_lalib_list[$indx]}
        elif [ "$b_t" = "acml" ]
        then
            lalib=${acml_lalib_list[$indx]}
        fi

        #config Makefile using a Python script
        python ../makefile_config.py $comp_label \
		`pwd` `which mpicc` ${mplib_list[$indx]} "$ladir" "$lalib" \
		$out_makef_name $b_t ${compilers_alias[$indx]} # > /dev/null

        #compile
        make arch=$comp_label > ../compile_out.txt
        echo "Compilation successful!"

        #config HPL.dat file with a Python script
        #for next call: PARAMS = {# cores, arch target name}
        mv bin/$comp_label/HPL.dat bin/$comp_label/HPL_buff.dat
        python ../hpl_dat_config.py $nr_cores $comp_label
        rm bin/$comp_label/HPL_buff.dat

        #execute!
        echo "Executing xhpl now:"
        cd "bin/"$comp_label
        #execute!
        mpirun ./xhpl
        cd ../../
        echo "...done executing xhpl."

        #clean current compilation
        make arch=$comp_label clean > ../compile_out.txt
        rm -R bin/
        rm -R lib/

        #TODO: add 'module list' output to the botton of the makefile

        cd ../
        mv hpl-2.1/$out_makef_name makefiles/nr_cores_$nr_cores/$out_makef_name

        rm -R hpl-2.1/

        #unload all modules
        module purge

        echo
        echo
    done

    ((indx++))
done

#cleaning overall files
./clean.sh
