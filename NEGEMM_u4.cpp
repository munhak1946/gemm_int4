#include <arm_neon.h>
#include "header/NEGEMM_u4.h"

#include "header/MatrixMap.h"
#define BLK_A      8
#define BLK_B      32

bool BLK_STATUS;
int partial_width;
int4_t *const_cast_ina_ptr_end_addr;
int4_t *const_cast_inb_ptr_end_addr;

template <typename T>
void matrix_matrix_multiply_s4(const MatrixMap<T> *ina, const MatrixMap<T> *inb, MatrixMap<int16_t> *out, float alpha) {

    const int ina_width = ina->col;
    const int ina_height = ina->row;
    const int ina_stride = ina->col /2;

    const int inb_width = inb->col;
    const int inb_height = inb->row;
    const int inb_stride = inb->col /2;

    const int out_width = out->col;
    const int out_height = out->row;

    const int out_stride = out->col;
    const int num_elems_matrix_b_x = inb->col;
    const int num_elems_matrix_b_y = inb->row;
    const int num_elems_matrix_b = num_elems_matrix_b_x * num_elems_matrix_b_y;


    const bool multiply_alpha = !is_one(alpha);

    const int8x16_t alpha_f16 = vdupq_n_s8(alpha);

    {
        const auto *mtx_a0 = reinterpret_cast<const int8_t *>(const_cast<MatrixMap <T> *>(ina)->ptr());
        const auto *mtx_b0 = reinterpret_cast<const int8_t *>(const_cast<MatrixMap <T> *>(inb)->ptr());
        auto *mtx_out = reinterpret_cast<int16_t *>(out->ptr());

        const auto *const_cast_ina_ptr_end_addr_int8_t = reinterpret_cast<const int8_t *>(const_cast_ina_ptr_end_addr);
        const auto *const_cast_inb_ptr_end_addr_int8_t = reinterpret_cast<const int8_t *>(const_cast_inb_ptr_end_addr);

        int16x8x8_t c_left00 =
                {{vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f)}};
        int16x8x8_t c_left01 =
                {{vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f)}};
        int16x8x8_t c_right00 =
                {{vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f)}};
        int16x8x8_t c_right01 =
                {{vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f),vdupq_n_s16(0.f)}};

        const int8x16_t andop = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15};

        for (; mtx_b0 <= (const_cast_inb_ptr_end_addr_int8_t) || (!BLK_STATUS && mtx_b0 < const_cast_inb_ptr_end_addr_int8_t);) {

            const int8x16_t p00 = vld1q_s8(mtx_a0);

            const int8x16_t q00 = vld1q_s8(mtx_b0);
            const int8x16_t q02 = vld1q_s8(mtx_b0 + inb_stride);
            const int8x16_t q04 = vld1q_s8(mtx_b0 + inb_stride * 2);
            const int8x16_t q06 = vld1q_s8(mtx_b0 + inb_stride * 3);


            int8x16_t p00_left = vshrq_n_s8(p00, 4);

            int8x16_t q00_left = vshrq_n_s8(q00, 4);
            int8x16_t q02_left = vshrq_n_s8(q02, 4);
            int8x16_t q04_left = vshrq_n_s8(q04, 4);
            int8x16_t q06_left = vshrq_n_s8(q06, 4);


            int8x16_t p00_right = vandq_s8(p00, andop);


            int8x16_t q00_right = vandq_s8(q00, andop);
            int8x16_t q02_right = vandq_s8(q02, andop);
            int8x16_t q04_right = vandq_s8(q04, andop);
            int8x16_t q06_right = vandq_s8(q06, andop);

            // if((!BLK_STATUS && mtx_b0 < const_cast_inb_ptr_end_addr_int8_t))
            //     printf("INB col left over block\n\n");
            // else
            //     printf("INB main block\n\n");


            c_left00.val[0] = vaddq_s16(c_left00.val[0], vmovl_s8(vget_low_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 0))))));
            c_left01.val[0] = vaddq_s16(c_left01.val[0], vmovl_s8(vget_high_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  0))))));
            c_left00.val[2] = vaddq_s16(c_left00.val[2], vmovl_s8(vget_low_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 1))))));
            c_left01.val[2] = vaddq_s16(c_left01.val[2], vmovl_s8(vget_high_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  1))))));
            c_left00.val[4] = vaddq_s16(c_left00.val[4], vmovl_s8(vget_low_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 2))))));
            c_left01.val[4] = vaddq_s16(c_left01.val[4], vmovl_s8(vget_high_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  2))))));
            c_left00.val[6] = vaddq_s16(c_left00.val[6], vmovl_s8(vget_low_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 3))))));
            c_left01.val[6] = vaddq_s16(c_left01.val[6], vmovl_s8(vget_high_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  3))))));

            c_left00.val[0] = vaddq_s16(c_left00.val[0], vmovl_s8(vget_low_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 4))))));
            c_left01.val[0] = vaddq_s16(c_left01.val[0], vmovl_s8(vget_high_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  4))))));
            c_left00.val[2] = vaddq_s16(c_left00.val[2], vmovl_s8(vget_low_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 5))))));
            c_left01.val[2] = vaddq_s16(c_left01.val[2], vmovl_s8(vget_high_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  5))))));
            c_left00.val[4] = vaddq_s16(c_left00.val[4], vmovl_s8(vget_low_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 6))))));
            c_left01.val[4] = vaddq_s16(c_left01.val[4], vmovl_s8(vget_high_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  6))))));
            c_left00.val[6] = vaddq_s16(c_left00.val[6], vmovl_s8(vget_low_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 7))))));
            c_left01.val[6] = vaddq_s16(c_left01.val[6], vmovl_s8(vget_high_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  7))))));

            c_left00.val[0] = vaddq_s16(c_left00.val[0], vmovl_s8(vget_low_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 8))))));
            c_left01.val[0] = vaddq_s16(c_left01.val[0], vmovl_s8(vget_high_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  8))))));
            c_left00.val[2] = vaddq_s16(c_left00.val[2], vmovl_s8(vget_low_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 9))))));
            c_left01.val[2] = vaddq_s16(c_left01.val[2], vmovl_s8(vget_high_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  9))))));
            c_left00.val[4] = vaddq_s16(c_left00.val[4], vmovl_s8(vget_low_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 10))))));
            c_left01.val[4] = vaddq_s16(c_left01.val[4], vmovl_s8(vget_high_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  10))))));
            c_left00.val[6] = vaddq_s16(c_left00.val[6], vmovl_s8(vget_low_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 11))))));
            c_left01.val[6] = vaddq_s16(c_left01.val[6], vmovl_s8(vget_high_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  11))))));

            c_left00.val[0] = vaddq_s16(c_left00.val[0], vmovl_s8(vget_low_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 12))))));
            c_left01.val[0] = vaddq_s16(c_left01.val[0], vmovl_s8(vget_high_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  12))))));
            c_left00.val[2] = vaddq_s16(c_left00.val[2], vmovl_s8(vget_low_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 13))))));
            c_left01.val[2] = vaddq_s16(c_left01.val[2], vmovl_s8(vget_high_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  13))))));
            c_left00.val[4] = vaddq_s16(c_left00.val[4], vmovl_s8(vget_low_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 14))))));
            c_left01.val[4] = vaddq_s16(c_left01.val[4], vmovl_s8(vget_high_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  14))))));
            c_left00.val[6] = vaddq_s16(c_left00.val[6], vmovl_s8(vget_low_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_left, 15))))));
            c_left01.val[6] = vaddq_s16(c_left01.val[6], vmovl_s8(vget_high_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_left,  15))))));

            //////////////////////////////////////////////////p00_left /////////////////////////////////////////////////////////////////


            c_right00.val[0] = vaddq_s16(c_right00.val[0], vmovl_s8(vget_low_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 0))))));
            c_right01.val[0] = vaddq_s16(c_right01.val[0], vmovl_s8(vget_high_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  0))))));
            c_right00.val[2] = vaddq_s16(c_right00.val[2], vmovl_s8(vget_low_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 1))))));
            c_right01.val[2] = vaddq_s16(c_right01.val[2], vmovl_s8(vget_high_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  1))))));
            c_right00.val[4] = vaddq_s16(c_right00.val[4], vmovl_s8(vget_low_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 2))))));
            c_right01.val[4] = vaddq_s16(c_right01.val[4], vmovl_s8(vget_high_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  2))))));
            c_right00.val[6] = vaddq_s16(c_right00.val[6], vmovl_s8(vget_low_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 3))))));
            c_right01.val[6] = vaddq_s16(c_right01.val[6], vmovl_s8(vget_high_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  3))))));

            c_right00.val[0] = vaddq_s16(c_right00.val[0], vmovl_s8(vget_low_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 4))))));
            c_right01.val[0] = vaddq_s16(c_right01.val[0], vmovl_s8(vget_high_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  4))))));
            c_right00.val[2] = vaddq_s16(c_right00.val[2], vmovl_s8(vget_low_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 5))))));
            c_right01.val[2] = vaddq_s16(c_right01.val[2], vmovl_s8(vget_high_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  5))))));
            c_right00.val[4] = vaddq_s16(c_right00.val[4], vmovl_s8(vget_low_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 6))))));
            c_right01.val[4] = vaddq_s16(c_right01.val[4], vmovl_s8(vget_high_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  6))))));
            c_right00.val[6] = vaddq_s16(c_right00.val[6], vmovl_s8(vget_low_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 7))))));
            c_right01.val[6] = vaddq_s16(c_right01.val[6], vmovl_s8(vget_high_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  7))))));

            c_right00.val[0] = vaddq_s16(c_right00.val[0], vmovl_s8(vget_low_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 8))))));
            c_right01.val[0] = vaddq_s16(c_right01.val[0], vmovl_s8(vget_high_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  8))))));
            c_right00.val[2] = vaddq_s16(c_right00.val[2], vmovl_s8(vget_low_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 9))))));
            c_right01.val[2] = vaddq_s16(c_right01.val[2], vmovl_s8(vget_high_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  9))))));
            c_right00.val[4] = vaddq_s16(c_right00.val[4], vmovl_s8(vget_low_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 10))))));
            c_right01.val[4] = vaddq_s16(c_right01.val[4], vmovl_s8(vget_high_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  10))))));
            c_right00.val[6] = vaddq_s16(c_right00.val[6], vmovl_s8(vget_low_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 11))))));
            c_right01.val[6] = vaddq_s16(c_right01.val[6], vmovl_s8(vget_high_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  11))))));
         

            c_right00.val[0] = vaddq_s16(c_right00.val[0], vmovl_s8(vget_low_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 12))))));
            c_right01.val[0] = vaddq_s16(c_right01.val[0], vmovl_s8(vget_high_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  12))))));
            c_right00.val[2] = vaddq_s16(c_right00.val[2], vmovl_s8(vget_low_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 13))))));
            c_right01.val[2] = vaddq_s16(c_right01.val[2], vmovl_s8(vget_high_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  13))))));
            c_right00.val[4] = vaddq_s16(c_right00.val[4], vmovl_s8(vget_low_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 14))))));
            c_right01.val[4] = vaddq_s16(c_right01.val[4], vmovl_s8(vget_high_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  14))))));
            c_right00.val[6] = vaddq_s16(c_right00.val[6], vmovl_s8(vget_low_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_left, 15))))));
            c_right01.val[6] = vaddq_s16(c_right01.val[6], vmovl_s8(vget_high_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_left,  15))))));


            //////////////////////////////////////////////////p00_left q_right /////////////////////////////////////////////////////////////////



            c_right00.val[1] = vaddq_s16(c_right00.val[1], vmovl_s8(vget_low_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 0))))));
            c_right01.val[1] = vaddq_s16(c_right01.val[1], vmovl_s8(vget_high_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 0))))));
            c_right00.val[3] = vaddq_s16(c_right00.val[3], vmovl_s8(vget_low_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 1))))));
            c_right01.val[3] = vaddq_s16(c_right01.val[3], vmovl_s8(vget_high_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 1))))));
            c_right00.val[5] = vaddq_s16(c_right00.val[5], vmovl_s8(vget_low_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 2))))));
            c_right01.val[5] = vaddq_s16(c_right01.val[5], vmovl_s8(vget_high_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 2))))));
            c_right00.val[7] = vaddq_s16(c_right00.val[7], vmovl_s8(vget_low_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 3))))));
            c_right01.val[7] = vaddq_s16(c_right01.val[7], vmovl_s8(vget_high_s8(vmulq_s8(q00_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 3))))));

            c_right00.val[1] = vaddq_s16(c_right00.val[1], vmovl_s8(vget_low_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_right,  4))))));
            c_right01.val[1] = vaddq_s16(c_right01.val[1], vmovl_s8(vget_high_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 4))))));
            c_right00.val[3] = vaddq_s16(c_right00.val[3], vmovl_s8(vget_low_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_right,  5))))));
            c_right01.val[3] = vaddq_s16(c_right01.val[3], vmovl_s8(vget_high_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 5))))));
            c_right00.val[5] = vaddq_s16(c_right00.val[5], vmovl_s8(vget_low_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_right,  6))))));
            c_right01.val[5] = vaddq_s16(c_right01.val[5], vmovl_s8(vget_high_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 6))))));
            c_right00.val[7] = vaddq_s16(c_right00.val[7], vmovl_s8(vget_low_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_right,  7))))));
            c_right01.val[7] = vaddq_s16(c_right01.val[7], vmovl_s8(vget_high_s8(vmulq_s8(q02_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 7))))));

            c_right00.val[1] = vaddq_s16(c_right00.val[1], vmovl_s8(vget_low_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_right,  8))))));
            c_right01.val[1] = vaddq_s16(c_right01.val[1], vmovl_s8(vget_high_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 8))))));
            c_right00.val[3] = vaddq_s16(c_right00.val[3], vmovl_s8(vget_low_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_right,  9))))));
            c_right01.val[3] = vaddq_s16(c_right01.val[3], vmovl_s8(vget_high_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 9))))));
            c_right00.val[5] = vaddq_s16(c_right00.val[5], vmovl_s8(vget_low_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_right,  10))))));
            c_right01.val[5] = vaddq_s16(c_right01.val[5], vmovl_s8(vget_high_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 10))))));
            c_right00.val[7] = vaddq_s16(c_right00.val[7], vmovl_s8(vget_low_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_right,  11))))));
            c_right01.val[7] = vaddq_s16(c_right01.val[7], vmovl_s8(vget_high_s8(vmulq_s8(q04_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 11))))));

            c_right00.val[1] = vaddq_s16(c_right00.val[1], vmovl_s8(vget_low_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 12))))));
            c_right01.val[1] = vaddq_s16(c_right01.val[1], vmovl_s8(vget_high_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 12))))));
            c_right00.val[3] = vaddq_s16(c_right00.val[3], vmovl_s8(vget_low_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 13))))));
            c_right01.val[3] = vaddq_s16(c_right01.val[3], vmovl_s8(vget_high_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 13))))));
            c_right00.val[5] = vaddq_s16(c_right00.val[5], vmovl_s8(vget_low_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 14))))));
            c_right01.val[5] = vaddq_s16(c_right01.val[5], vmovl_s8(vget_high_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 14))))));
            c_right00.val[7] = vaddq_s16(c_right00.val[7], vmovl_s8(vget_low_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 15))))));
            c_right01.val[7] = vaddq_s16(c_right01.val[7], vmovl_s8(vget_high_s8(vmulq_s8(q06_right, vdupq_n_s8(vgetq_lane_s8(p00_right, 15))))));

            //////////////////////////////////////////////////p00_right /////////////////////////////////////////////////////////////////


            c_left00.val[1] = vaddq_s16(c_left00.val[1], vmovl_s8(vget_low_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  0))))));
            c_left01.val[1] = vaddq_s16(c_left01.val[1], vmovl_s8(vget_high_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 0))))));
            c_left00.val[3] = vaddq_s16(c_left00.val[3], vmovl_s8(vget_low_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  1))))));
            c_left01.val[3] = vaddq_s16(c_left01.val[3], vmovl_s8(vget_high_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 1))))));
            c_left00.val[5] = vaddq_s16(c_left00.val[5], vmovl_s8(vget_low_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  2))))));
            c_left01.val[5] = vaddq_s16(c_left01.val[5], vmovl_s8(vget_high_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 2))))));
            c_left00.val[7] = vaddq_s16(c_left00.val[7], vmovl_s8(vget_low_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  3))))));
            c_left01.val[7] = vaddq_s16(c_left01.val[7], vmovl_s8(vget_high_s8(vmulq_s8(q00_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 3))))));

            c_left00.val[1] = vaddq_s16(c_left00.val[1], vmovl_s8(vget_low_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  4))))));
            c_left01.val[1] = vaddq_s16(c_left01.val[1], vmovl_s8(vget_high_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 4))))));
            c_left00.val[3] = vaddq_s16(c_left00.val[3], vmovl_s8(vget_low_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  5))))));
            c_left01.val[3] = vaddq_s16(c_left01.val[3], vmovl_s8(vget_high_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 5))))));
            c_left00.val[5] = vaddq_s16(c_left00.val[5], vmovl_s8(vget_low_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  6))))));
            c_left01.val[5] = vaddq_s16(c_left01.val[5], vmovl_s8(vget_high_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 6))))));
            c_left00.val[7] = vaddq_s16(c_left00.val[7], vmovl_s8(vget_low_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  7))))));
            c_left01.val[7] = vaddq_s16(c_left01.val[7], vmovl_s8(vget_high_s8(vmulq_s8(q02_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 7))))));

            c_left00.val[1] = vaddq_s16(c_left00.val[1], vmovl_s8(vget_low_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  8))))));
            c_left01.val[1] = vaddq_s16(c_left01.val[1], vmovl_s8(vget_high_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 8))))));
            c_left00.val[3] = vaddq_s16(c_left00.val[3], vmovl_s8(vget_low_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  9))))));
            c_left01.val[3] = vaddq_s16(c_left01.val[3], vmovl_s8(vget_high_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 9))))));
            c_left00.val[5] = vaddq_s16(c_left00.val[5], vmovl_s8(vget_low_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  10))))));
            c_left01.val[5] = vaddq_s16(c_left01.val[5], vmovl_s8(vget_high_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 10))))));
            c_left00.val[7] = vaddq_s16(c_left00.val[7], vmovl_s8(vget_low_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  11))))));
            c_left01.val[7] = vaddq_s16(c_left01.val[7], vmovl_s8(vget_high_s8(vmulq_s8(q04_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 11))))));

            c_left00.val[1] = vaddq_s16(c_left00.val[1], vmovl_s8(vget_low_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  12))))));
            c_left01.val[1] = vaddq_s16(c_left01.val[1], vmovl_s8(vget_high_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 12))))));
            c_left00.val[3] = vaddq_s16(c_left00.val[3], vmovl_s8(vget_low_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  13))))));
            c_left01.val[3] = vaddq_s16(c_left01.val[3], vmovl_s8(vget_high_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 13))))));
            c_left00.val[5] = vaddq_s16(c_left00.val[5], vmovl_s8(vget_low_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  14))))));
            c_left01.val[5] = vaddq_s16(c_left01.val[5], vmovl_s8(vget_high_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 14))))));
            c_left00.val[7] = vaddq_s16(c_left00.val[7], vmovl_s8(vget_low_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_right,  15))))));
            c_left01.val[7] = vaddq_s16(c_left01.val[7], vmovl_s8(vget_high_s8(vmulq_s8(q06_left, vdupq_n_s8(vgetq_lane_s8(p00_right, 15))))));

            //////////////////////////////////////////////////p00_right q left /////////////////////////////////////////////////////////////////

            mtx_a0 += 16;
            mtx_b0 += (4 * inb_stride );

        }


        /*   if(multiply_alpha)
           {
               c.val[0] = vmulq_f16(c.val[0], alpha_f16);
               c.val[1] = vmulq_f16(c.val[1], alpha_f16);
               c.val[2] = vmulq_f16(c.val[2], alpha_f16);
               c.val[3] = vmulq_f16(c.val[3], alpha_f16);
           }
           */

        int16x8x2_t lr00 = vzipq_s16(c_left00.val[0], c_right00.val[0]); int16x8x2_t lr01 = vzipq_s16(c_left00.val[1], c_right00.val[1]);
        int16x8x2_t lr02 = vzipq_s16(c_left00.val[2], c_right00.val[2]); int16x8x2_t lr03 = vzipq_s16(c_left00.val[3], c_right00.val[3]);
        int16x8x2_t lr04 = vzipq_s16(c_left00.val[4], c_right00.val[4]); int16x8x2_t lr05 = vzipq_s16(c_left00.val[5], c_right00.val[5]);
        int16x8x2_t lr06 = vzipq_s16(c_left00.val[6], c_right00.val[6]); int16x8x2_t lr07 = vzipq_s16(c_left00.val[7], c_right00.val[7]);


        int16x8x2_t lr10 = vzipq_s16(c_left01.val[0], c_right01.val[0]); int16x8x2_t lr11 = vzipq_s16(c_left01.val[1], c_right01.val[1]);
        int16x8x2_t lr12 = vzipq_s16(c_left01.val[2], c_right01.val[2]); int16x8x2_t lr13 = vzipq_s16(c_left01.val[3], c_right01.val[3]);
        int16x8x2_t lr14 = vzipq_s16(c_left01.val[4], c_right01.val[4]); int16x8x2_t lr15 = vzipq_s16(c_left01.val[5], c_right01.val[5]);
        int16x8x2_t lr16 = vzipq_s16(c_left01.val[6], c_right01.val[6]); int16x8x2_t lr17 = vzipq_s16(c_left01.val[7], c_right01.val[7]);

        int id_y = 0;
        int id_x = 0;

        if (BLK_STATUS) {

            if (id_x < (out_width - 8)) {
                vst1q_s16(mtx_out, lr00.val[0]);        vst1q_s16(mtx_out + 8, lr00.val[1]);
                vst1q_s16(mtx_out + 16, lr10.val[0]);   vst1q_s16(mtx_out + 24, lr10.val[1]);

                if (id_y + 1 < out_height) {
                    vst1q_s16(mtx_out + 1 * out_stride, lr01.val[0]);       vst1q_s16(mtx_out + 1 * out_stride + 8, lr01.val[1]);
                    vst1q_s16(mtx_out + 1 * out_stride + 16, lr11.val[0]);  vst1q_s16(mtx_out + 1 * out_stride + 24, lr11.val[1]);

                    if (id_y + 2 < out_height) {
                        vst1q_s16(mtx_out + 2 * out_stride, lr02.val[0]);           vst1q_s16(mtx_out + 2 * out_stride + 8, lr02.val[1]);
                        vst1q_s16(mtx_out + 2 * out_stride + 16, lr12.val[0]);      vst1q_s16(mtx_out + 2 * out_stride + 24, lr12.val[1]);


                        if (id_y + 3 < out_height) {
                            vst1q_s16(mtx_out + 3 * out_stride, lr03.val[0]);       vst1q_s16(mtx_out + 3 * out_stride + 8, lr03.val[1]);
                            vst1q_s16(mtx_out + 3 * out_stride + 16, lr13.val[0]);  vst1q_s16(mtx_out + 3 * out_stride + 24, lr13.val[1]);

                            if (id_y + 4 < out_height) {
                                vst1q_s16(mtx_out + 4 * out_stride, lr04.val[0]);           vst1q_s16(mtx_out + 4 * out_stride + 8, lr04.val[1]);
                                vst1q_s16(mtx_out + 4 * out_stride + 16, lr14.val[0]);      vst1q_s16(mtx_out + 4 * out_stride + 24, lr14.val[1]);

                                if (id_y + 5 < out_height) {
                                    vst1q_s16(mtx_out + 5 * out_stride, lr05.val[0]);       vst1q_s16(mtx_out + 5 * out_stride + 8, lr05.val[1]);
                                    vst1q_s16(mtx_out + 5 * out_stride + 16, lr15.val[0]);  vst1q_s16(mtx_out + 5 * out_stride + 24, lr15.val[1]);

                                    if (id_y + 6 < out_height) {
                                        vst1q_s16(mtx_out + 6 * out_stride, lr06.val[0]);       vst1q_s16(mtx_out + 6 * out_stride + 8, lr06.val[1]);
                                        vst1q_s16(mtx_out + 6 * out_stride + 16, lr16.val[0]);  vst1q_s16(mtx_out + 6 * out_stride + 24, lr16.val[1]);

                                        if (id_y + 7 < out_height) {
                                            vst1q_s16(mtx_out + 7 * out_stride, lr07.val[0]);       vst1q_s16(mtx_out + 7 * out_stride + 8, lr07.val[1]);
                                            vst1q_s16(mtx_out + 7 * out_stride + 16, lr17.val[0]);  vst1q_s16(mtx_out + 7 * out_stride + 24, lr17.val[1]);


                                        }

                                    }
                                }

                            }
                        }
                    }
                }
            }

        } else {
            // printf("\ninb col left over store\n");

            //printf("inb col left over %d partial_width = %d\n", lr00.val[0][1],partial_width);
            if (id_x < 8) {
                memcpy(mtx_out,      &lr00.val[0], (partial_width  >=  8 ? 8 : partial_width     ) * sizeof(int16_t));
                memcpy(mtx_out + 8,  &lr00.val[1], (partial_width  >= 16 ? 8 : (partial_width -   8 < 0 ? 0 : partial_width -   8)) * sizeof(int16_t));
                memcpy(mtx_out + 16, &lr10.val[0], (partial_width  >= 24 ? 8 : (partial_width -  16 < 0 ? 0 : partial_width -  16)) * sizeof(int16_t));
                memcpy(mtx_out + 24, &lr10.val[1], (partial_width  >= 32 ? 8 : (partial_width -  24 < 0 ? 0 : partial_width -  24)) * sizeof(int16_t));

                if (id_y + 1 < out_height) {
                    memcpy(mtx_out + 1 * out_stride,      &lr01.val[0], (partial_width  >=  8 ? 8 : partial_width     ) * sizeof(int16_t));
                    memcpy(mtx_out + 1 * out_stride + 8,  &lr01.val[1], (partial_width  >= 16 ? 8 : (partial_width -   8 < 0 ? 0 : partial_width -   8)) * sizeof(int16_t));
                    memcpy(mtx_out + 1 * out_stride + 16, &lr11.val[0], (partial_width  >= 24 ? 8 : (partial_width -  16 < 0 ? 0 : partial_width -  16)) * sizeof(int16_t));
                    memcpy(mtx_out + 1 * out_stride + 24, &lr11.val[1], (partial_width  >= 32 ? 8 : (partial_width -  24 < 0 ? 0 : partial_width -  24)) * sizeof(int16_t));


                    if (id_y + 2 < out_height) {
                        memcpy(mtx_out + 2 * out_stride,      &lr02.val[0], (partial_width  >=  8 ? 8 : partial_width     ) * sizeof(int16_t));
                        memcpy(mtx_out + 2 * out_stride + 8,  &lr02.val[1], (partial_width  >= 16 ? 8 : (partial_width -   8 < 0 ? 0 : partial_width -   8)) * sizeof(int16_t));
                        memcpy(mtx_out + 2 * out_stride + 16, &lr12.val[0], (partial_width  >= 24 ? 8 : (partial_width -  16 < 0 ? 0 : partial_width -  16)) * sizeof(int16_t));
                        memcpy(mtx_out + 2 * out_stride + 24, &lr12.val[1], (partial_width  >= 32 ? 8 : (partial_width -  24 < 0 ? 0 : partial_width -  24)) * sizeof(int16_t));

                        if (id_y + 3 < out_height) {
                            memcpy(mtx_out + 3 * out_stride,      &lr03.val[0], (partial_width  >=  8 ? 8 : partial_width     ) * sizeof(int16_t));
                            memcpy(mtx_out + 3 * out_stride + 8,  &lr03.val[1], (partial_width  >= 16 ? 8 : (partial_width -   8 < 0 ? 0 : partial_width -   8)) * sizeof(int16_t));
                            memcpy(mtx_out + 3 * out_stride + 16, &lr13.val[0], (partial_width  >= 24 ? 8 : (partial_width -  16 < 0 ? 0 : partial_width -  16)) * sizeof(int16_t));
                            memcpy(mtx_out + 3 * out_stride + 24, &lr13.val[1], (partial_width  >= 32 ? 8 : (partial_width -  24 < 0 ? 0 : partial_width -  24)) * sizeof(int16_t));

                            if (id_y + 4 < out_height) {
                                memcpy(mtx_out + 4 * out_stride,      &lr04.val[0], (partial_width  >=  8 ? 8 : partial_width     ) * sizeof(int16_t));
                                memcpy(mtx_out + 4 * out_stride + 8,  &lr04.val[1], (partial_width  >= 16 ? 8 : (partial_width -   8 < 0 ? 0 : partial_width -   8)) * sizeof(int16_t));
                                memcpy(mtx_out + 4 * out_stride + 16, &lr14.val[0], (partial_width  >= 24 ? 8 : (partial_width -  16 < 0 ? 0 : partial_width -  16)) * sizeof(int16_t));
                                memcpy(mtx_out + 4 * out_stride + 24, &lr14.val[1], (partial_width  >= 32 ? 8 : (partial_width -  24 < 0 ? 0 : partial_width -  24)) * sizeof(int16_t));

                                if (id_y + 5 < out_height) {
                                    memcpy(mtx_out + 5 * out_stride,      &lr05.val[0], (partial_width  >=  8 ? 8 : partial_width     ) * sizeof(int16_t));
                                    memcpy(mtx_out + 5 * out_stride + 8,  &lr05.val[1], (partial_width  >= 16 ? 8 : (partial_width -   8 < 0 ? 0 : partial_width -   8)) * sizeof(int16_t));
                                    memcpy(mtx_out + 5 * out_stride + 16, &lr15.val[0], (partial_width  >= 24 ? 8 : (partial_width -  16 < 0 ? 0 : partial_width -  16)) * sizeof(int16_t));
                                    memcpy(mtx_out + 5 * out_stride + 24, &lr15.val[1], (partial_width  >= 32 ? 8 : (partial_width -  24 < 0 ? 0 : partial_width -  24)) * sizeof(int16_t));


                                    if (id_y + 6 < out_height) {
                                        memcpy(mtx_out + 6 * out_stride,      &lr06.val[0], (partial_width  >=  8 ? 8 : partial_width     ) * sizeof(int16_t));
                                        memcpy(mtx_out + 6 * out_stride + 8,  &lr06.val[1], (partial_width  >= 16 ? 8 : (partial_width -   8 < 0 ? 0 : partial_width -   8)) * sizeof(int16_t));
                                        memcpy(mtx_out + 6 * out_stride + 16, &lr16.val[0], (partial_width  >= 24 ? 8 : (partial_width -  16 < 0 ? 0 : partial_width -  16)) * sizeof(int16_t));
                                        memcpy(mtx_out + 6 * out_stride + 24, &lr16.val[1], (partial_width  >= 32 ? 8 : (partial_width -  24 < 0 ? 0 : partial_width -  24)) * sizeof(int16_t));

                                        if (id_y + 7 < out_height) {
                                            memcpy(mtx_out + 7 * out_stride,      &lr07.val[0], (partial_width  >=  8 ? 8 : partial_width     ) * sizeof(int16_t));
                                            memcpy(mtx_out + 7 * out_stride + 8,  &lr07.val[1], (partial_width  >= 16 ? 8 : (partial_width -   8 < 0 ? 0 : partial_width -   8)) * sizeof(int16_t));
                                            memcpy(mtx_out + 7 * out_stride + 16, &lr17.val[0], (partial_width  >= 24 ? 8 : (partial_width -  16 < 0 ? 0 : partial_width -  16)) * sizeof(int16_t));
                                            memcpy(mtx_out + 7 * out_stride + 24, &lr17.val[1], (partial_width  >= 32 ? 8 : (partial_width -  24 < 0 ? 0 : partial_width -  24)) * sizeof(int16_t));

                                        }

                                    }
                                }

                            }
                        }
                    }
                }
            }
        }

    }
    return;

}

//#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
