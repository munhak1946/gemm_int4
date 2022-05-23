//
// Created by eslab on 2021-07-16.
//

#ifndef GEMM_MATRIXMAP_H
#define GEMM_MATRIXMAP_H

template <typename T>
class MatrixMap{
public:
    T* input;
    int row, col;
    MatrixMap(T* _input, int _row, int _col);
    T* ptr();
    ~MatrixMap();

};

#endif //GEMM_MATRIXMAP_H
