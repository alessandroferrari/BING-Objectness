#include <string>    
#include<boost/python.hpp>
#include "py_liblinear.hpp"

using namespace boost::python;

BOOST_PYTHON_MODULE(py_liblinear)
{
	class_<PyLibLinear>("PyLibLinear", init<>())
	      .def(init<>())
	      .def("matRead", &PyLibLinear::matRead)
		  .def("matWrite", &PyLibLinear::matWrite)
		  .def("trainSVM", &PyLibLinear::trainSVM)
	    ;
}
