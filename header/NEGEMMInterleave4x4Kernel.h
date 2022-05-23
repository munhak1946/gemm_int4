//
// Created by eslab on 2021-07-16.
//
#include "MatrixMap.h"

#ifndef GEMM_NEGEMMINTERLEAVE4X4KERNEL_H
#define GEMM_NEGEMMINTERLEAVE4X4KERNEL_H

template <typename T>
void gemm_interleave4x4(const MatrixMap<T>* in, MatrixMap<T>* out);

template <typename T>
void run(const MatrixMap<T> *_input, MatrixMap<T> *_output);

#endif //GEMM_NEGEMMINTERLEAVE4X4KERNEL_H
