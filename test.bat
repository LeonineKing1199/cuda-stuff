@echo off
call vars.bat
cuda-memcheck .\%build_directory%\regulus_tests.exe %1