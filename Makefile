
ifndef CPPC
	CPPC=g++
endif

CPP_COMMON = Cpp_common

CCFLAGS=

INC = -I$(CPP_COMMON) -I./

LIBS = -lOpenCL -lrt
MAIN_SRCS = main.cpp mnist.cpp
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

all: main data_transfer

main:
	$(CPPC) $(MAIN_SRCS) $(INC) $(CCFLAGS) $(LIBS) -o main

data_transfer:
	$(CPPC) $(DATA_SRCS) $(INC) $(CCFLAGS) $(LIBS) -o data_transfer


clean:
	rm -f main data_transfer *.o
