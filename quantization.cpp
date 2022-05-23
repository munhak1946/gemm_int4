//
// Created by eslab on 2021-03-23.
//
#include <arm_neon.h>
#include <iostream>
#include <cstring>
#include "header/MatrixMap.h"
#include "MatrixMap.cpp"

using namespace std;
/*void uniform_midtread_quantizer(x, Q){

    xQ = np.round(x / Q) * Q
    return xQ

}
void LaplacianClippingAnalysis(Alpha, b,bitWidth){
    Analysis = []
    for(alpha in Alpha;;) {
        mse = 2 * (b ** 2) * ((np.e) ** (-alpha / b)) + ((alpha ** 2) / (3 * (2 ** (2 * bitWidth))))
        Analysis.append(mse)

    }
    return Analysis

}

void LaplacianClippingSimulation(Alpha, b, bitWidth){
    simulations = []
    highPrecision = np.random.laplace(scale=b, size=100000, loc = 0)
    for(alpha in Alpha;;){

            s = np.copy(highPrecision)
            Q = (2*alpha)/(2**bitWidth)

//#clipping
            s[s > alpha ] = alpha
            s[s < -alpha] = -alpha
//# quantization
            s = uniform_midtread_quantizer(s, Q)

            mse = ((s - highPrecision) ** 2).mean()
            simulations.append(mse)
    }
    return simulations

}*/

void FindMax(const MatrixMap<float> lhs, float* lhs_max, const MatrixMap<float> rhs, float* rhs_max) {
    *lhs_max = lhs.input[0];
    for (int i = 0; i < lhs.row * lhs.col; i++) {
        const float val0 = lhs.input[i];
        *lhs_max = std::max(*lhs_max, val0);
    }

    *rhs_max = rhs.input[0];
    for (int i = 0; i < rhs.row * rhs.col; i++) {
        const float val1 = rhs.input[i];
        *rhs_max = std::max(*rhs_max, val1);
    }
}

void Quantize(const MatrixMap<float> lhs, float lhs_max, const MatrixMap<float> rhs, float rhs_max){
    float clipping_value = 5.03;

    for(int i =0; i < lhs.col * lhs.row ; i+=4){
        float32x4_t input = vld1q_f32(lhs.input + i);
        vst1q_f32(lhs.input + i, vmulq_n_f32(input, clipping_value / lhs_max));
    }

    for(int i =0; i < rhs.col * rhs.row  ; i+=4){
        float32x4_t input = vld1q_f32(rhs.input + i);
        vst1q_f32(rhs.input + i, vmulq_n_f32(input, clipping_value / rhs_max));

    }


}

void Quantize_Int4(float* input0, float* input1, int M, int N, int K){
    float lhs_max, rhs_max;
    MatrixMap<float> lhs(input0, M, K);
    MatrixMap<float> rhs(input1, K, N);

    FindMax(lhs, &lhs_max, rhs, &rhs_max);

    Quantize(lhs, lhs_max, rhs, rhs_max);
    for(int i = 0; i < M ;i ++){
        for(int j = 0; j < K ; j++)
            if( i == j )
                cout << *(input0 + i*M + j) <<", " <<*(input1 + i*M + j) <<endl;


    }


}