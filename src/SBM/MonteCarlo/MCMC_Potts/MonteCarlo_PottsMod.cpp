#include <iostream>
#include <cmath>
#include <random>
#include <ctime>
#include <vector>
#include <omp.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace std;

const char* MC_doc = "Doc";
PyObject* MC(PyObject*,PyObject* args) {
	PyArrayObject *wO, *StatesO;
	int tburn,Q;

	if (!PyArg_ParseTuple(args, "O!O!ii",
		&PyArray_Type, &wO,
		&PyArray_Type, &StatesO,
		&tburn,
		&Q)) {
			PyErr_SetString(PyExc_RuntimeError, "Failed to parse input");
			return nullptr;
		}

	const int N = PyArray_DIM(StatesO, 0);
    const int L = PyArray_DIM(StatesO, 1);
    const double* w = (double*) PyArray_DATA(wO);
    int* States = (int*) PyArray_DATA(StatesO);
    const unsigned int seed0 = time(nullptr);

    omp_set_num_threads(50);
    #pragma omp parallel
    {
		std::uniform_int_distribution<int>unifpos(0,L-1);
		std::uniform_real_distribution<double>unifrate(0.,1.);
		int thread_id = omp_get_thread_num();
		std::mt19937 rng(seed0 + thread_id);

		//const unsigned int seed = seed0+123u*omp_get_thread_num();
		//default_random_engine re(seed);

		#pragma omp for
		for (int m = 0; m < N; m++) {
			for (int k = 0; k < tburn; k++) {

				const int pos = unifpos(rng);
				int cur_aa = States[m * L + pos];
				int dq = 1 + static_cast<int>(unifrate(rng) * (Q - 1));
				int new_aa = (cur_aa + dq) % Q;

				double dE = w[(L * (L - 1) / 2 * Q * Q + pos * Q + new_aa)] -
							w[(L * (L - 1) / 2 * Q * Q + pos * Q + cur_aa)];

				for (int other_pos = pos + 1; other_pos < L; other_pos++) {
					dE += w[Q * Q * ((pos * (L * 2 - pos - 1) / 2 + (other_pos - pos - 1))) + new_aa * Q + States[m * L + other_pos]] -
							w[Q * Q * ((pos * (L * 2 - pos - 1) / 2 + (other_pos - pos - 1))) + cur_aa * Q + States[m * L + other_pos]];
				}

				for (int other_pos = 0; other_pos < pos; other_pos++) {
					dE += w[Q * Q * ((other_pos * (L * 2 - other_pos - 1) / 2 + (pos - other_pos - 1))) + States[m * L + other_pos] * Q + new_aa] -
							w[Q * Q * ((other_pos * (L * 2 - other_pos - 1) / 2 + (pos - other_pos - 1))) + States[m * L + other_pos] * Q + cur_aa];
				}

				if (dE >= 0. || unifrate(rng) < exp(dE)) {
					States[m * L + pos] = new_aa;
				}
			}
		}
    }

    return Py_BuildValue("");
}
static PyMethodDef MonteCarlo_methods[] = {
	{ "MC", (PyCFunction)MC, METH_VARARGS, MC_doc },
	{ nullptr, nullptr, 0, nullptr }
};

static PyModuleDef MonteCarlo_module = {
	PyModuleDef_HEAD_INIT,
	"MonteCarlo_Potts",  // Module name to use with Python
	"Potts MCMC implemented in c++",  // Module description
	0,
	MonteCarlo_methods  // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_MonteCarlo_Potts() {
	import_array();
    return PyModule_Create(&MonteCarlo_module);
}
