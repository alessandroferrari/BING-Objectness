PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

BOOST_INC = /usr/include
BOOST_LIB = /usr/lib
OPENCV_LIB = $$(pkg-config --libs opencv)
OPENCV_INC = $$(pkg-config --cflags opencv)
NUMPY_INCLUDE = /usr/local/lib/python$(PYTHON_VERSION)/dist-packages/numpy/core/include

TARGET = filter_tig
SRC = filter_tig.cpp FilterTIG.cpp
OBJ = filter_tig.o FilterTIG.o

$(TARGET).so: $(OBJ)
	g++ -shared $(OBJ) -L$(BOOST_LIB) -lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config -lpython$(PYTHON_VERSION) -o $(TARGET).so $(OPENCV_LIB)

$(OBJ): $(SRC)
	g++ -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -I$(BOOST_INC) $(OPENCV_CFLAGS) -fPIC -c -std=c++0x $(SRC)

clean:
	rm -f $(OBJ)
	rm -f $(TARGET).so
