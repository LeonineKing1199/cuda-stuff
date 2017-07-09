@echo off
call vars.bat
cuda-memcheck .\build\%configuration_type%\regulus_tests.exe -s