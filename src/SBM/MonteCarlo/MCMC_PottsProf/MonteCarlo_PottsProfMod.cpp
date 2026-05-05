#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <omp.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace std;

// MC(w, states, tburn, Q, seed) -> None
// Profile-only variant: w packs only fields h (no couplings J).
// See MCMC_Potts/MonteCarlo_PottsMod.cpp for full ABI notes.
const char* MC_doc = "Run Metropolis MCMC on a Potts profile model. See ABI in source.";
PyObject* MC(PyObject*,PyObject* args) {
	PyArrayObject *wO, *StatesO;
	int tburn, Q;
	unsigned long long seed0;

	if (!PyArg_ParseTuple(args, "O!O!iiK",
		&PyArray_Type, &wO,
		&PyArray_Type, &StatesO,
		&tburn,
		&Q,
		&seed0)) {
			PyErr_SetString(PyExc_RuntimeError, "Failed to parse input");
			return nullptr;
		}

	const int N = PyArray_DIM(StatesO, 0);
    const int L = PyArray_DIM(StatesO, 1);
    const double* w = (double*) PyArray_DATA(wO);
    int* States = (int*) PyArray_DATA(StatesO);

    #pragma omp parallel
    {
		std::uniform_int_distribution<int>unifpos(0,L-1);
		std::uniform_real_distribution<double>unifrate(0.,1.);
		int thread_id = omp_get_thread_num();
		std::mt19937_64 rng(seed0 + static_cast<uint64_t>(thread_id));

		#pragma omp for
		for (int m = 0; m < N; m++) {
			for (int k = 0; k < tburn; k++) {

				const int pos = unifpos(rng);
				int cur_aa = States[m * L + pos];
				int dq = 1 + static_cast<int>(unifrate(rng) * (Q - 1));
				int new_aa = (cur_aa + dq) % Q;

				double dE = w[(pos * Q + new_aa)] -
							w[(pos * Q + cur_aa)];

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
	"MonteCarlo_PottsProf",  // Module name to use with Python
	"Potts MCMC implemented in c++",  // Module description
	0,
	MonteCarlo_methods  // Structure that defines the methods of the module
};

PyMODINIT_FUNC PyInit_MonteCarlo_PottsProf() {
	import_array();
    return PyModule_Create(&MonteCarlo_module);
}
