#!/bin/bash

#TODO:
#	for initial tests, implement this in such a way that
#	after executing './xhpl', the PID of the associated
#	process is saved, and then, after 1 min, that process
#	is killed. Send all outputs to some file. To do this tests,
#	submit a job without the bash file, and perform the tests

#TODO:
#	check the specific RAM specs (maybe submit a job for 5 min
#	and check RAM specs there)

#Execution instructions
#	$ ./general.sh NR_CORES

nr_cores=$1

#list of compilers
compilers=("gcc/4.9.3-gnu" "intel/cc/64/11.0.074" "intel/12.1/composer_xe_2011_sp1.9.293" "intel/15.0.6/composer_xe_2015.6.233")
compilers_labels=("gcc" "intel11" "intel12" "intel15")

#parallel compiler support
par_comp_support=("default-gcc-openmpi" "default-intel-openmpi" "openmpi/1.6.5-intel12.1.3" "intel/15.0.6/impi-5.0.3.049")
par_comp_labels=("openmpi" "openmpi" "openmpi" "impi")

#no need to load MKL for last two
blas_support_mkl=("mkl/64/11.0.074" "mkl/64/11.0.074" " " " ")
blas_support_acml=("acml/64/intel/4.3.0" "acml/64/intel/4.3.0" "acml/64/intel/4.3.0" "acml/64/intel/4.3.0")

mplib_list=("libmpi.so" "libmpi.so" "libmpi.so" "libmpich.a")

#TODO: complete next line
mkl_ladir_list=("/home/support/apps/intel/cc/64/11.0.074/mkl/lib/em64t" \
	"/home/support/apps/intel/cc/64/11.0.074/mkl/lib/em64t" " " " ")
#TODO: complete next line
mkl_lalib_list=("-lmkl_intel_lp64 -lmkl_sequential -lmkl_core" "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core" \
	" " " ")

#TODO: next line complete
acml_ladir_list=()
#TODO: next line complete
acml_lalib_list=()

#make dir to output Makefiles
mkdir makefiles

#first copy of Makefile
cp setup/Make.Linux_PII_CBLAS Make.lonsdale_gcc_mkl

blas_types=("mkl" "acml")

#For each compiler, do one compilation and execution for
#both MKL and ACML
indx=0
for comp in ${compilers[@]}
do
    module load ${compilers[$indx]} ${openmpi_support[$indx]}
    for b_t in ${blas_types[@]}
    do
    
        out_makef_name="Make.lonsdale_"${compilers_labels[$indx]}"_"${par_comp_labels[$indx]}"_"$b_t
    
        #load modules
        if [ "$b_t" = "mkl" ]
        then
            module load ${blas_support_mkl[$indx]}
        elif [ "$b_t" = "acml" ]
            module load ${blas_support_acml[$indx]}
        fi

        if [ "$b_t" = "mkl" ]
        then
            ladir=${mkl_ladir_list[$indx]}
        elif [ "$b_t" = "acml" ]
            ladir=${acml_ladir_list[$indx]}
        fi
        
        if [ "$b_t" = "mkl" ]
        then
            lalib=${mkl_lalib_list[$indx]}
        elif [ "$b_t" = "acml" ]
            lalib=${acml_lalib_list[$indx]}
                python makefile_config.py "lonsdale_"${compilers_labels[$indx]}"_"${par_comp_labels[$indx]}"_"$b_t \


fi        #TODO: config Makefile (complete Python script)
		`pwd` `which mpicc` ${mplib_list[@]} $ladir $lalib \
		$out_makef_name $b_t

        #compile #TODO: uncomment next line!
        #make arch="lonsdale_"${compilers_labels[$indx]}"_"${par_comp_labels[$indx]}"_"$b_t

        #TODO: config HPL.dat file (create a Python file for this as well!)
        #python hpl_dat_config.py PARAMS

        #TODO: execute!
        cd "bin/lonsdale_"${compilers_labels[$indx]}"_"${par_comp_labels[$indx]}"_"$b_t
        #execute
        cd ../../

        make arch=lonsdale_gcc_mkl clean
        
        #TODO: add 'module list' output to the botton of the makefile

        mv $out_makef_name makefiles/$out_makef_name$nr_cores
        
        #unload modules
        if [ "$b_t" = "mkl" ]
        then
            module unload ${blas_support_mkl[$indx]}
        elif [ "$b_t" = "acml" ]
            module unload ${blas_support_acml[$indx]}
        fi
        
        #TODO: play with next line for tests!
        exit
    done
    
    #unload previous modules
    module unload ${compilers[$indx]} ${openmpi_support[$indx]}
    
    ((indx++))
done
