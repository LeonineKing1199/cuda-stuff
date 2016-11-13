nvcc = nvcc
nvcc_flags = -c --expt-extended-lambda \
             -gencode arch=compute_50,code=sm_50 \
             -rdc=true -O3 -std=c++11

nvlink = nvlink

obj_dir = bin
project = regulus.exe

objects = \
  main.obj \
  \
  src^/point.obj \
  src^/peano.obj \
  src^/nominate.obj \
  src^/fract-locations.obj \
  src^/fracture.obj \
  src^/get-assoc-size.obj \
  src^/rand-int-range.obj \
  src^/mark-nominated-tetra.obj \
  \
  tests^/test-suite.obj \
  tests^/domain-tests.obj \
  tests^/mesher-tests.obj \
  tests^/math-tests.obj \
  tests^/matrix-tests.obj \
  tests^/array-tests.obj \
  tests^/tetra-tests.obj \
  tests^/nomination-tests.obj \
  tests^/fract-location-tests.obj \
  tests^/fracture-tests.obj \
  tests^/redistribute-pts-tests.obj \
  tests^/stack-vector-tests.obj \
  tests^/get-assoc-size-tests.obj

executable = $(obj_dir)\$(project)

all: $(executable)

$(executable): $(objects)
	$(nvcc) -gencode arch=compute_50,code=sm_50 -lcudadevrt $** -o $@

$(objects):
	$(nvcc) $(nvcc_flags) $*.cu -o $@

clean: $(objects)
	del /q $(objects)

