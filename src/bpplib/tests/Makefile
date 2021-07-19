CPP=g++17
CFLAGS=-Wall -O3 -std=c++17 -m64 -msse4.2 -fopenmp
INCLUDE=../include #/opt/intel/tbb/include
LDFLAGS=-fopenmp
LIBS=rt pthread
LIBDIRS=
#LIBDIRS=/opt/intel/tbb/lib/intel64/cc4.1.0_libc2.4_kernel2.6.16.21
SOURCES=$(shell find . -name '*.cpp')
HEADERS=$(shell find . -name '*.hpp') $(shell find ../include -name '*.hpp')
OBJS=$(patsubst ./%,./.objs/%,$(SOURCES:%.cpp=%.o))
OBJDIRS=$(shell find . -type d | grep -vE 'svn|[.]objs' | sed 's/^[.]/.\/.objs/')
EXECUTABLE=./tests


.PHONY: all clear clean purge

all: $(EXECUTABLE)


# Calculate dependencies...

Makefile.dep: $(SOURCES) $(HEADERS)
	@echo Calculating dependencies...
	@$(CPP) $(CFLAGS) -MM $(addprefix -I,$(INCLUDE)) $(SOURCES) > $@

-include Makefile.dep



# Building Targets

$(EXECUTABLE): $(OBJDIRS) $(OBJS)
	@echo Linking executable \'"$@"\' ...
	@$(CPP) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) $(OBJS) -o "$@"

$(OBJDIRS):
	@mkdir -p "$@"

.objs/%.o: ./%.cpp
	@echo Compiling \'"$@"\' ...
	@$(CPP) -c $(CFLAGS) $(addprefix -I,$(INCLUDE)) "$<" -o "$@"



# Cleaning Stuff

clear:
	@echo Removing object files...
	-@rm -Rf ./.objs
	-@rm -f $(EXECUTABLE)

clean: clear

purge: clear
	@echo Removing all generated files...
	-@rm -f $(EXECUTABLE)
	-@rm -f Makefile.dep

