#include <Python.h>
#include <string>
class PyLibLinear
{
    public:
    PyLibLinear();
    void matWrite(std::string filename, PyObject* arr);
    PyObject* matRead(std::string filename);
    PyObject* trainSVM(PyObject* X1f, PyObject* Y, int sT, double C, double bias = -1, double eps = 0.0001);
};
