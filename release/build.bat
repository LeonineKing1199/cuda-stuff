mkdir ..\build-release
cd    ..\build-release

cmake ^
    -G "Ninja" ^
    -DCMAKE_BUILD_TYPE=Release ^
    ..\release\ 

ninja -j 4

cd ..\release