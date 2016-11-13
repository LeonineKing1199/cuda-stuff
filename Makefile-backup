nvcc=nvcc
nvcc_flags=-c -std=c++11 -O3 \
           --compiler-options -Wall \
           --compiler-options -Wextra \
           --compiler-options -Wno-unused-parameter \
           -gencode arch=compute_50,code=sm_50 \
           --expt-extended-lambda \
           -D_FORCE_INLINES \
           -rdc=true

ldflags=-lcudadevrt

sources= \
  main.cu \
  \
  src/point.cu \
  src/peano.cu \
  src/nominate.cu \
  src/fract-locations.cu \
  src/fracture.cu \
  src/get-assoc-size.cu \
  src/rand-int-range.cu \
  src/mark-nominated-tetra.cu \
  \
  tests/test-suite.cu \
  tests/domain-tests.cu \
  tests/mesher-tests.cu \
  tests/math-tests.cu \
  tests/matrix-tests.cu \
  tests/array-tests.cu \
  tests/tetra-tests.cu \
  tests/nomination-tests.cu \
  tests/fract-location-tests.cu \
  tests/fracture-tests.cu \
  tests/redistribute-pts-tests.cu \
  tests/stack-vector-tests.cu \
  tests/get-assoc-size-tests.cu

obj_dir=build
objects=$(addsuffix .o, $(addprefix $(obj_dir)/, $(basename $(sources))))
executable=$(obj_dir)/regulus


.PHONY:
all:  $(executable)

$(executable): $(objects)
	$(nvcc) $(ldflags) $(objects) -o $@

$(obj_dir)/%.o: %.cu
	mkdir -p $(dir $@)
	$(nvcc) $(nvcc_flags) $< -o $@

.PHONY: clean
clean:
	rm -r $(obj_dir)/*
