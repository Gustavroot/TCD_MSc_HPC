#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include "ranlxd.h"
//#define LVEC 12


static PyObject *genrand(PyObject *self, PyObject *args)
{
    int Nrandnrs;
    //int sts;

    //From the args, the number N is taken
    if (!PyArg_ParseTuple(args, "i", &Nrandnrs))
        return NULL;
    double *randarray = (double *) malloc(Nrandnrs);
    //Obtaining the array of random numbers from RANLUX
    ranlxd(randarray,Nrandnrs);

    printf("blar\n");
    return Py_BuildValue("i", 5);
}


static PyMethodDef PyranluxMethods[] = {
 { "genrand", genrand, METH_VARARGS, "Use RANLUX" },
 { NULL, NULL, 0, NULL }
};


PyMODINIT_FUNC

initpyranlux(void)
{
  (void) Py_InitModule("pyranlux", PyranluxMethods);
}
