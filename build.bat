@echo off

call vars.bat

mkdir %build_directory% 2> NUL
cd %build_directory%

set CC=cl
set CXX=cl

cmake ^
  -DCMAKE_TOOLCHAIN_FILE=%cmake_toolchain_file% ^
  -DCMAKE_BUILD_TYPE=%configuration_type% ^
  -G "Ninja" ^
  .. &&^
cmake --build .

cd ..