//
// Created by eslab on 2021-03-23.
//
#include "header/MatrixMap.h"

template <typename T>
MatrixMap<T>::MatrixMap(T* _input, int _row, int _col) : input(_input), row(_row), col(_col){}

template <typename T>
T* MatrixMap<T>::ptr(){
    return this.input;
}

template <typename T>
MatrixMap<T>::~MatrixMap(){}
