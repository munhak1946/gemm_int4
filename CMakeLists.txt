cmake_minimum_required(VERSION 3.16)
project(GEMM)

set(CMAKE_CXX_STANDARD 14)
add_executable(GEMM main.cpp header/MatrixMap.h header/NEGEMMInterleave4x4Kernel.h header/Type.h header/NEGEMM_u4.h)