PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

BOOST_INC = /usr/include
BOOST_LIB = /usr/lib
OPENCV_LIB = $$(pkg-config --libs opencv)
OPENCV_INC = $$(pkg-config --cflags opencv)
NUMPY_INCLUDE = /usr/local/lib/python$(PYTHON_VERSION)/dist-packages/numpy/core/include
LIBLINEAR_DIR = $(ROOT_DIR)/LibLinear
LIBLINEAR_INCLUDE = $(LIBLINEAR_DIR)

TARGET = liblinear
SRC = py_liblinear.cpp liblinear_wrapper.cpp 
OBJ = py_liblinear.o liblinear_wrapper.o

$(TARGET).so: $(OBJ)
	g++ -shared $(OBJ) -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so ./LibLinear/liblinear.a $(OPENCV_LIB)

$(OBJ): $(SRC)
	g++ -I$(PYTHON_INCLUDE) -I$(LIBLINEAR_INCLUDE) -I$(BOOST_INC) -I$(NUMPY_INCLUDE) $(OPENCV_CFLAGS) -fPIC -c -std=c++0x $(SRC)

clean:
	rm -f $(OBJ)
	rm -f $(TARGET).so
