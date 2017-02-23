mkdir ..\build-debug
cd    ..\build-debug

cmake ^
    -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Debug ^
    ..\debug\ 

cmake --build .

cd ..\debug