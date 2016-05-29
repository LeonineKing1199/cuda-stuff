################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../test/math.cu \
../test/orientation.cu 

OBJS += \
./test/math.o \
./test/orientation.o 

CU_DEPS += \
./test/math.d \
./test/orientation.d 


# Each subdirectory must supply rules for building sources it contributes
test/%.o: ../test/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -O3 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "test" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -O3 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


