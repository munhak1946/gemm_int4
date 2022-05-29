//
// Created by eslab on 2021-07-19.
//
#include "MatrixMap.h"
#include <cmath>

#ifndef GEMM_NEGEMM_U4_H
#define GEMM_NEGEMM_U4_H


bool is_one(float a, float epsilon = 0.00001f){  return std::abs(1.0f - a) <= epsilon; }

template <typename T>
void matrix_matrix_multiply_s4(const MatrixMap<T> *ina, const MatrixMap<T> *inb, MatrixMap<int16_t> *out, float alpha);

template <typename T>
void run_matrix_matrix_multiply_s4(const MatrixMap<T> *ina, const MatrixMap<T> *inb, MatrixMap<int16_t> *out, float alpha);


#endif //GEMM_NEGEMM_U4_H
