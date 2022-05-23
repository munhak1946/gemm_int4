#include <iostream>
#include <arm_neon.h>
#include "header/MatrixMap.h"
#include "header/Type.h"
#include <typeinfo>

using namespace std;
uint8x16_t double_elements(	uint8x16_t input)

{

    return(vmulq_u8(input, input));

}
template <typename T>
char* function(T a){
    return const_cast<char*>(typeid(a).name());
}
int main() {
/*
    uint8_t a[] = {1,2,3,4,5,6,7,8, 9,10,11,12,13,14,15,16};
    uint8_t b[] = {1,2,3,4,5,6,7,8, 1,2,3,4,1,2,3,4};
    printf("%u\n", a[0]);
    uint8x16_t res_add = vaddq_u8(vld1q_u8(a),vld1q_u8(b));
    uint8x16_t res_mul = double_elements(vld1q_u8(a));
    printf("%u\n", res_add[15]);
    printf("%u\n", res_mul[15]);*/
/*
    cout << vgetq_lane_u8(double_elements(vld1q_u8(a)),0) <<endl;
    double_elements(a);

    float16x4_t b = double_elements(vld1_f16(a));
    cout << vget_lane_u16(b,1) << endl;
    cout << res[0] << endl;*/


    float16x8x4_t c =
            {
                    {
                            vdupq_n_f16(0.f),
                            vdupq_n_f16(0.f),
                            vdupq_n_f16(0.f),
                            vdupq_n_f16(0.f)
                    }
            };

    float16_t mtx_a0[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float16_t mtx_b0[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    const float16x8_t p00 = vld1q_f16(mtx_a0);
    const float16x8_t p02 = vld1q_f16(mtx_a0 + 8);


    const float16x8_t q00 = vld1q_f16(mtx_b0);
    const float16x8_t q02 = vld1q_f16(mtx_b0 + 8);

    vgetq_lane_f16(p00, 0);
    vld1q_f16(mtx_b0);
//    vmulq_n_f16(vld1q_f16(mtx_b0), vgetq_lane_f16(p00, 0));
//    cout << q00[0] << endl;
//    vmulq_n_f16(q00, vgetq_lane_f16(p00, 0));
//    c.val[0] = vaddq_f16(c.val[0], vmulq_n_f16(vld1q_f16(mtx_b0), vgetq_lane_f16(p00, 0)));


    int8x8_t ai= {1,2,3,4,5,6,7,8};
    int8x8_t bi= {9,10,11,12,13,14,15,16};

    int8x8_t ci = vhadd_s8(ai,bi);

    int8x8x2_t abinterleave = vzip_s8(ai,bi);

    int16x8_t ai16= {1,2,3,4,5,6,7,8};

    uint8x8_t ciu = vqshlu_n_s8 (ai,2);
    int8x8_t ciu2 = vshl_n_s8 (ai,2);

    for(int i =0 ;i < 16; i++)
        printf("%d\n", abinterleave.val[1][i]);

    cout << "Hello, World?" << endl;
    return 0;
}
