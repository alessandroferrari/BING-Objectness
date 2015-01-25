#include<boost/python.hpp>
#include "FilterTIG.hpp"

using namespace boost::python;

BOOST_PYTHON_MODULE(filter_tig)
{
	class_<FilterTIG>("FilterTIG", init<>())
	      .def(init<>())
	      .def("update", &FilterTIG::update)
		  .def("match_template", &FilterTIG::matchTemplate)
		  .def("reconstruct", &FilterTIG::reconstruct)
		  .def("non_maxima_suppression", &FilterTIG::nonMaxSup)
	    ;
}
