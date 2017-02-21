mkdir ..\build-debug
cd    ..\build-debug

cmake ^
    -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Debug ^
    ..\debug\ 

ninja -j 4

cd ..\debug