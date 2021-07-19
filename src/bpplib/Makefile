.PHONY: all clear clean purge tests

all:
	make -C tests all
	make -C cuda_tests all
	
clean:
	make -C tests clean
	make -C cuda_tests clean
	
clear:
	make -C tests clear
	make -C cuda_tests clear
	
purge:
	make -C tests purge
	make -C cuda_tests purge
	
test: all
	tests/tests
	cuda_tests/cuda_tests
	
