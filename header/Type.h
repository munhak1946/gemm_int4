//
// Created by eslab on 2021-07-16.
//

#ifndef GEMM_TYPE_H
#define GEMM_TYPE_H

typedef struct _int4{
    unsigned int value : 4 ;
}int4_t;

typedef struct _ScalarType{
    int4_t value;
}ScalarType;

#endif //GEMM_TYPE_H
