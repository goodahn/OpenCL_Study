
ifndef CPPC
	CPPC=g++
endif

CPP_COMMON = Cpp_common

CCFLAGS=

INC = -I$(CPP_COMMON) -I./

LIBS = -lOpenCL -lrt
MATRIX_SRCS = matrix_main.cpp
MAIN_SRCS = mnist_main.cpp mnist.cpp
DATA_SRCS = data_transfer.cpp
TARGET = out

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CPPC = clang++
	CCFLAGS += -stdlib=libc++
	LIBS = -framework OpenCL
endif

CCFLAGS += -D DEVICE=$(DEVICE)

all: matrix_main mnist_main data_transfer

matrix_main:
	$(CPPC) $(MATRIX_SRCS) $(INC) $(CCFLAGS) $(LIBS) -o matrix_main

data_transfer:
	$(CPPC) $(DATA_SRCS) $(INC) $(CCFLAGS) $(LIBS) -o data_transfer

mnist_main:
	$(CPPC) $(MAIN_SRCS) $(INC) $(CCFLAGS) $(LIBS) -o mnist_main


clean:
	rm -f mnist_main data_transfer matrix_main *.o
