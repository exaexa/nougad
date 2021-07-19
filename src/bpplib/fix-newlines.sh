#!/bin/bash

dos2unix ./include/*.hpp ./include/*/*.hpp ./include/*/*/*.hpp ./include/*/*/*/*.hpp
dos2unix ./tests/*.hpp ./tests/*.cpp ./tests/*/*.cpp ./tests/*/*/*.cpp ./tests/*/*/*/*.cpp
dos2unix ./cuda_tests/*.hpp ./cuda_tests/*.cpp ./cuda_tests/*/*.cpp ./cuda_tests/*/*/*.cpp ./cuda_tests/*/*/*/*.cpp
