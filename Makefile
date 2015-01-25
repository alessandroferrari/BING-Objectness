ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/source/cpp

#compiling blas static library

BLAS_DIR = $(ROOT_DIR)/custom_liblinear_wrapper/LibLinear/blas

AR     = ar rcv
RANLIB = ranlib 

BLAS_SOURCES = $(BLAS_DIR)/dnrm2.c $(BLAS_DIR)/daxpy.c $(BLAS_DIR)/ddot.c $(BLAS_DIR)/dscal.c
BLAS_HEADERS = $(BLAS_DIR)/blas.h $(BLAS_DIR)/blasp.h
BLAS_OBJ = dnrm2.o daxpy.o ddot.o dscal.o 

CFLAGS = $(OPTFLAGS) 
FFLAGS = $(OPTFLAGS)

BLAS_TARGET = blas
LIBLINEAR_TARGET = liblinear
PY_LIBLINEAR_TARGET = py_liblinear
FTIG_TARGET = filter_tig

all: $(BLAS_TARGET).a $(LIBLINEAR_TARGET).a $(PY_LIBLINEAR_TARGET).so $(FTIG_TARGET).so install clean

$(BLAS_TARGET).a: $(BLAS_OBJ) $(BLAS_HEADERS)
	$(AR) blas.a $(BLAS_OBJ)  
	$(RANLIB) blas.a
$(BLAS_OBJ):
	$(CC) $(CFLAGS) -fPIC -c $(BLAS_SOURCES)

#compiling liblinear static library

LIBLINEAR_DIR = $(ROOT_DIR)/custom_liblinear_wrapper/LibLinear
LIBLINEAR_INCLUDE = $(LIBLINEAR_DIR)

BLAS_INCLUDE = $(BLAS_DIR)

LIBLINEAR_SRC = $(LIBLINEAR_DIR)/train.c $(LIBLINEAR_DIR)/linear.cpp $(LIBLINEAR_DIR)/tron.cpp 
LIBLINEAR_OBJ = train.o linear.o tron.o

$(LIBLINEAR_TARGET).a: $(LIBLINEAR_OBJ)
	#g++ -shared $(OBJ)  -o $(TARGET).so
	$(AR) $(LIBLINEAR_TARGET).a $(LIBLINEAR_OBJ) $(BLAS_OBJ)
	$(RANLIB) $(LIBLINEAR_TARGET).a
	
$(LIBLINEAR_OBJ): $(LIBLINEAR_SRC)
	g++ -I$(BLAS_DIR) -fPIC -c -std=c++0x $(LIBLINEAR_SRC) 

#some python/opencv/numpy definition

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

BOOST_INC = /usr/include
BOOST_LIB = /usr/lib
OPENCV_LIB = $$(pkg-config --libs opencv)
OPENCV_INC = $$(pkg-config --cflags opencv)
NUMPY_INCLUDE = /usr/local/lib/python$(PYTHON_VERSION)/dist-packages/numpy/core/include

#now compiling python liblinear wrapper

PY_LIBLINEAR_DIR = $(ROOT_DIR)/custom_liblinear_wrapper

PY_LIBLINEAR_SRC = $(PY_LIBLINEAR_DIR)/py_liblinear.cpp $(PY_LIBLINEAR_DIR)/liblinear_wrapper.cpp 
PY_LIBLINEAR_OBJ = py_liblinear.o liblinear_wrapper.o

$(PY_LIBLINEAR_TARGET).so: $(PY_LIBLINEAR_OBJ)
	g++ -shared -L$(BOOST_LIB) $(PY_LIBLINEAR_OBJ) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(PY_LIBLINEAR_TARGET).so $(OPENCV_LIB) liblinear.a

$(PY_LIBLINEAR_OBJ): $(PY_LIBLINEAR_SRC)
	g++ -I$(PYTHON_INCLUDE) -I$(BLAS_INCLUDE) -I$(LIBLINEAR_INCLUDE) -I$(BOOST_INC) -I$(NUMPY_INCLUDE) $(OPENCV_CFLAGS) -fPIC -c -std=c++0x $(PY_LIBLINEAR_SRC)

#compiling fast bing filtering
FTIG_DIR = $(ROOT_DIR)/filter_tig_source
FTIG_SRC = $(FTIG_DIR)/filter_tig.cpp $(FTIG_DIR)/FilterTIG.cpp
FTIG_OBJ = filter_tig.o FilterTIG.o

$(FTIG_TARGET).so: $(FTIG_OBJ)
	g++ -shared $(FTIG_OBJ) -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(FTIG_TARGET).so $(OPENCV_LIB)

$(FTIG_OBJ): $(FTIG_SRC)
	g++ -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -I$(BOOST_INC) $(OPENCV_CFLAGS) -fPIC -c -std=c++0x $(FTIG_SRC)

install:
	mkdir -p build
	cp *.so build
	cp *.a build

clean:
	- rm -f *.o
	- rm -f *.a
	- rm -f *~
	- rm -f *.so