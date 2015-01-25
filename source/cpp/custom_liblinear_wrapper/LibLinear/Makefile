ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
AR     = ar rcv
RANLIB = ranlib

BLAS_DIR = $(ROOT_DIR)/blas
BLAS_INCLUDE = $(BLAS_DIR)

TARGET = liblinear
SRC = linear.cpp train.c tron.cpp 
OBJ = linear.o train.o tron.o
BLAS_OBJ = $(BLAS_DIR)/dnrm2.o $(BLAS_DIR)/daxpy.o $(BLAS_DIR)/ddot.o $(BLAS_DIR)/dscal.o

$(TARGET).a: $(OBJ)
	#g++ -shared $(OBJ)  -o $(TARGET).so
	$(AR) $(TARGET).a $(OBJ) $(BLAS_OBJ)
	$(RANLIB) $(TARGET).a
	
$(OBJ): $(SRC)
	g++ -I$(BLAS_INCLUDE) -fPIC -c -std=c++0x $(SRC) 

clean:
	rm -f $(OBJ)
	rm -f $(TARGET).so
