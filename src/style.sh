#!/bin/sh
cd `dirname $0` #lol
clang-format -style="{BasedOnStyle: Mozilla, PointerAlignment: Right}" -verbose -i `find -name '*.c' -or -name '*.cpp' -or -name  '*.h' -or -name '*.hpp' -or -name '*.cu' -or -name '*.cuh'`
