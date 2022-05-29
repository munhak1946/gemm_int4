//
// Created by eslab on 2021-07-16.
//

#ifndef GEMM_TYPE_H
#define GEMM_TYPE_H

#include <arm_neon.h>
//
//typedef struct _int4{
//    uint8_t val : 4 ;
//
//}int4_t;

typedef struct _int4{
    int8_t val1 : 4 ;
    int8_t val0 : 4 ;

}int4_t;

typedef struct {
    int16x8_t val[8];
}int16x8x8_t;

#endif //GEMM_TYPE_H
