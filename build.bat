@echo off


call vars.bat

mkdir %build_directory% 2> NUL
cd %build_directory%

cmake ^
  -DCMAKE_TOOLCHAIN_FILE=%cmake_toolchain_file% ^
  -DCMAKE_BUILD_TYPE=%configuration_type% ^
  -G "Visual Studio 14 2015 Win64" ^
  .. &&^
cmake --build .

cd ..