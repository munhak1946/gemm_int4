//
// Created by eslab on 2021-03-23.
//
#include "MatrixMap.h"
#include "../MatrixMap.cpp"

#ifndef GEMM_INT4_QUANTIZATION_H
#define GEMM_INT4_QUANTIZATION_H
void FindMax(const MatrixMap<float> lhs, float* lhs_max, const MatrixMap<float> rhs, float* rhs_max);
void Quantize(MatrixMap<float> lhs, float lhs_max, MatrixMap<float> rhs, float rhs_max);
void Quantize_Int4(float* input0, float* input1, int M, int N, int K);
#endif //GEMM_INT4_QUANTIZATION_H
