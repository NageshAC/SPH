CUDA = nvcc
CPP = g++
CUDA_FLAGS = -O3 #-std=c++11 -aarch=sm_60
CPP_FLAGS = -O3 -std=c++11

SOURCES_PATH = ./src/
INPUT_PATH = ./data/
OUTPUT_PATH = ./sln/
INCLUDE_PATH = ./src/includes/

MAIN = $($(SOURCES_PATH) + "simulate.cu")
EXAMPLE = ./example/create_data.cpp
HEADERS = ./src/includes/check_status.hpp


all: example simulate clean

simulate: $(MAIN) $(HEADERS)
	$(CUDA) $(CUDA_FLAGS) ./src/simulate.cu -o simulate
	./simulate.exe

example: $(EXAMPLE)
	$(CPP)  $(CPP_FLAGS) ./example/create_data.cpp -o create_data
	./create_data.exe

clean: 
	rm -r *.exe *.exp *.lib