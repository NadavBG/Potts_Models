# Toolchain file for macOS to use Homebrew LLVM (supports OpenMP)

if(NOT APPLE)
    return()
endif()

# Detect default Homebrew installation path
set(HOMEBREW_LLVM "/opt/homebrew/opt/llvm" CACHE PATH "")

set(CMAKE_C_COMPILER "${HOMEBREW_LLVM}/bin/clang")
set(CMAKE_CXX_COMPILER "${HOMEBREW_LLVM}/bin/clang++")

# Ensure the runtime and include paths are picked up
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${HOMEBREW_LLVM}/include")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${HOMEBREW_LLVM}/lib")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${HOMEBREW_LLVM}/lib")

# OpenMP
set(OpenMP_CXX "${HOMEBREW_LLVM}/bin/clang++")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "${HOMEBREW_LLVM}/lib/libomp.dylib")
