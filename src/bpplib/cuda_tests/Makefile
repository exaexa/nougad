CPP=icpc
CFLAGS=-Wall -O2 -std=c++11 -msse4.2
CUDA_FLAGS=-O3
CUDA_ARCH=-arch sm_35
INCLUDE=../include /usr/local/cuda/include
LDFLAGS=
LIBS=rt cudart
LIBDIRS=/usr/local/cuda/lib64
SOURCES=$(shell find . -name '*.cpp')
CUDA_SOURCES=$(shell find . -name '*.cu')
HEADERS=$(shell find . -name '*.hpp') $(shell find ../include -name '*.hpp')
OBJS=$(patsubst ./%,./.objs/%,$(SOURCES:%.cpp=%.o))
CUDA_OBJS=$(patsubst ./%,./.objs/%,$(CUDA_SOURCES:%.cu=%.co))
OBJDIRS=$(shell find . -type d | grep -vE 'svn|[.]objs' | sed 's/^[.]/.\/.objs/')
EXECUTABLE=./cuda_tests


.PHONY: all clear clean purge

all: $(EXECUTABLE)


# Calculate dependencies...

Makefile.dep: $(SOURCES) $(HEADERS)
	@echo Calculating dependencies...
	@$(CPP) $(CFLAGS) -MM $(addprefix -I,$(INCLUDE)) $(SOURCES) > $@

-include Makefile.dep



# Building Targets

$(EXECUTABLE): $(OBJDIRS) $(OBJS) $(CUDA_OBJS)
	@echo Linking executable \'"$@"\' ...
	@$(CPP) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $(OBJS) $(CUDA_OBJS) -o "$@"


$(OBJDIRS):
	@mkdir -p "$@"


.objs/%.o: ./%.cpp
	@echo Compiling \'"$@"\' ...
	@$(CPP) -c $(CFLAGS) $(addprefix -I,$(INCLUDE)) "$<" -o "$@"

.objs/%.co: ./%.cu
	@echo Compiling CUDA file \'"$@"\' ...
	@nvcc $(CUDA_FLAGS) $(addprefix -I,$(INCLUDE)) --compile $(CUDA_ARCH) -cudart static $< -o $@   




# Cleaning Stuff

clear:
	@echo Removing object files...
	-@rm -Rf ./.objs

clean: clear

purge: clear
	@echo Removing all generated files...
	-@rm -f $(EXECUTABLE)
	-@rm -f Makefile.dep

