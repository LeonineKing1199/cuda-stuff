#!/bin/bash

# start up the "IDE"
# it's noisy so we silence it by piping everything to /dev/null
gedit ../main.cu ../CMakeLists.txt > /dev/null 2>&1 &

