################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../tests/domain-tests.cpp \
../tests/test-suite.cpp 

OBJS += \
./tests/domain-tests.o \
./tests/test-suite.o 

CPP_DEPS += \
./tests/domain-tests.d \
./tests/test-suite.d 


# Each subdirectory must supply rules for building sources it contributes
tests/%.o: ../tests/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O2 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "tests" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O2 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


