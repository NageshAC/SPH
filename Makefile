CUDA = nvcc
CUDA_PROFILER = nvprof
CPP = g++
CUDA_FLAGS = -O3 #-std=c++11 -aarch=sm_60
CPP_FLAGS = -O3 -std=c++11

SOURCES_PATH = ./src/
INPUT_PATH = ./data/
OUTPUT_PATH = ./sln/
INCLUDE_PATH = ./src/includes/

MAIN = $($(SOURCES_PATH) + "simulate.cu")
EXAMPLE = ./example/create_data.cpp
HEADERS = $($(SOURCES_PATH) + "check_status.cu") $($(SOURCES_PATH) + "parameters.cpp")  $($(SOURCES_PATH) + "particle.cpp")  $($(SOURCES_PATH) + "input.cpp")  $($(SOURCES_PATH) + "wireframe.cpp")  $($(SOURCES_PATH) + "vtk.cpp") $($(SOURCES_PATH) + "check.cu")


all: example simulate clean

simulate: $(MAIN) $(HEADERS)
	$(CUDA) $(CUDA_FLAGS) ./src/simulate.cu -o simulate
	./simulate.exe

example: $(EXAMPLE)
	$(CPP)  $(CPP_FLAGS) ./example/create_data.cpp -o create_data
	./create_data.exe

clean: 
	rm -r *.exe *.exp *.lib

profile: $(MAIN) $(HEADERS)
	$(CPP)  $(CPP_FLAGS) ./example/create_data.cpp -o create_data
	./create_data.exe
	$(CUDA) $(CUDA_FLAGS) ./src/simulate.cu -o simulate
	$(CUDA_PROFILER) ./simulate.exe
	rm -r *.exe *.exp *.lib
