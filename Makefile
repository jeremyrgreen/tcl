INCLUDE_FLAGS=-I./include/ -I${HPTT_ROOT}/include/

# BLIS
BLAS_LIB_DIR = ${BLIS_ROOT}/lib
BLAS_LIB=-L${BLAS_LIB_DIR} -lblis -lm -lpthread

# # MKL
# BLAS_LIB_DIR = ${MKLROOT}/lib/intel64
# BLAS_LIB=-L${BLAS_LIB_DIR} -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
# INCLUDE_FLAGS +=-I${MKLROOT}/include

# # OPENBLAS
# BLAS_LIB_DIR = ${OPENBLAS_ROOT}/lib
# BLAS_LIB=-L${BLAS_LIB_DIR} -lopenblas -lpthread
# INCLUDE_FLAGS +=-I${OPENBLAS_ROOT}/include

CXX_LINK=-L${HPTT_ROOT}/lib -lhptt ${BLAS_LIB} \
    -Wl,--enable-new-dtags,-rpath,${HPTT_ROOT}/lib \
    -Wl,--enable-new-dtags,-rpath,${BLAS_LIB_DIR}
CXX_FLAGS=-O3 -std=c++11 -fPIC ${INCLUDE_FLAGS} -fopenmp -march=native

scalar:
	${MAKE} clean
	${MAKE} scalar2

scalar2: all

SRC=$(wildcard ./src/*.cpp)
OBJ=$(SRC:.cpp=.o)

all: ${OBJ}
	mkdir -p lib
	${CXX} ${OBJ} ${CXX_FLAGS} -o lib/libtcl.so -shared ${CXX_LINK}
	ar rvs lib/libtcl.a ${OBJ}

%.o: %.cpp
	${CXX} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@

clean:
	rm -rf src/*.o lib/libtcl.so lib/libtcl.a
