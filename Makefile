CC = gcc
CCFLAGS=-O3 -Iinclude -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include"
LIBS = -Llib -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\Win32" -lOpenCL -fopenmp -lm


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
	LIBS = -framework OpenCL
endif

CCFLAGS += -D DEVICE=$(DEVICE)

main: ocl.c main.c
	$(CC) $^ $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f main
