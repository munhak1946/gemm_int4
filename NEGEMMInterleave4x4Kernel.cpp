#include <typeinfo>
#include <cstring>
#include "header/NEGEMMInterleave4x4Kernel.h"

#define B       4
#define BMEM    2

using namespace std;

template <typename T>
void gemm_interleave4x4(const MatrixMap<T>* in, MatrixMap<T>* out)
{
    auto const_cast_in      = const_cast<MatrixMap<T> *>(in);
    auto const_cast_in_ptr  = const_cast_in->ptr();
    const int in_stride     = const_cast_in->col / 2;
    const int in_width      = const_cast_in->col;
    const int in_height     = const_cast_in->row;

    const int block_width   = in_width / B;
    const int partial_width = in_width % B;

    int window_start_width = 0;
    int window_end_width = window_start_width + B;



    for(int block = 0; block < block_width; ++block){

        for(int x = window_start_width; x < window_end_width; ++x)
        {

            T data0[] =
                    {
                            {((const_cast_in_ptr +  1 * in_stride) + block * BMEM + x)->val0,
                             ((const_cast_in_ptr +  0 * in_stride) + block * BMEM + x)->val0},

                            {((const_cast_in_ptr +  3 * in_stride) + block * BMEM + x)->val0,
                             ((const_cast_in_ptr +  2 * in_stride) + block * BMEM + x)->val0},

                            {((const_cast_in_ptr +  5 * in_stride) + block * BMEM + x)->val0,
                             ((const_cast_in_ptr +  4 * in_stride) + block * BMEM + x)->val0},

                            {((const_cast_in_ptr +  7 * in_stride) + block * BMEM + x)->val0,
                             ((const_cast_in_ptr +  6 * in_stride) + block * BMEM + x)->val0},

                    };

            T data1[] =
                    {
                            {((const_cast_in_ptr +  1 * in_stride) + block * BMEM + x)->val1,
                             ((const_cast_in_ptr +  0 * in_stride) + block * BMEM + x)->val1},

                            {((const_cast_in_ptr +  3 * in_stride) + block * BMEM + x)->val1,
                             ((const_cast_in_ptr +  2 * in_stride) + block * BMEM + x)->val1},

                            {((const_cast_in_ptr +  5 * in_stride) + block * BMEM + x)->val1,
                             ((const_cast_in_ptr +  4 * in_stride) + block * BMEM + x)->val1},

                            {((const_cast_in_ptr +  7 * in_stride) + block * BMEM + x)->val1,
                             ((const_cast_in_ptr +  6 * in_stride) + block * BMEM + x)->val1},

                    };
            memcpy(out->ptr() + (x * 2)     * B + ( block * B * 4 ) , data0, B * sizeof(T) );
            memcpy(out->ptr() + (x * 2 + 1) * B + ( block * B * 4 ) , data1, B * sizeof(T) );

        }
    }

    for(int x = 0; x < partial_width; ++x){


            T data0[] =
                    {
                            {((const_cast_in_ptr +  1 * in_stride) + (block_width-1) * BMEM + x)->val0,
                             ((const_cast_in_ptr +  0 * in_stride) + (block_width-1) * BMEM + x)->val0},

                            {((const_cast_in_ptr +  3 * in_stride) + (block_width-1) * BMEM + x)->val0,
                             ((const_cast_in_ptr +  2 * in_stride) + (block_width-1) * BMEM + x)->val0},

                            {((const_cast_in_ptr +  5 * in_stride) + (block_width-1) * BMEM + x)->val0,
                             ((const_cast_in_ptr +  4 * in_stride) + (block_width-1) * BMEM + x)->val0},

                            {((const_cast_in_ptr +  7 * in_stride) + (block_width-1) * BMEM + x)->val0,
                             ((const_cast_in_ptr +  6 * in_stride) + (block_width-1) * BMEM + x)->val0},


                    };

            T data1[] =
                    {
                            {((const_cast_in_ptr +  1 * in_stride) + (block_width-1) * BMEM + x)->val1,
                             ((const_cast_in_ptr +  0 * in_stride) + (block_width-1) * BMEM + x)->val1},

                            {((const_cast_in_ptr +  3 * in_stride) + (block_width-1) * BMEM + x)->val1,
                             ((const_cast_in_ptr +  2 * in_stride) + (block_width-1) * BMEM + x)->val1},

                            {((const_cast_in_ptr +  5 * in_stride) + (block_width-1) * BMEM + x)->val1,
                             ((const_cast_in_ptr +  4 * in_stride) + (block_width-1) * BMEM + x)->val1},

                            {((const_cast_in_ptr +  7 * in_stride) + (block_width-1) * BMEM + x)->val1,
                             ((const_cast_in_ptr +  6 * in_stride) + (block_width-1) * BMEM + x)->val1},

                    };
            memcpy(out->ptr() + (x * 2)     * B + ( (block_width-1) * B * 4 ) , data0, B * sizeof(T) );
            memcpy(out->ptr() + (x * 2 + 1) * B + ( (block_width-1) * B * 4 ) , data1, B * sizeof(T) );


    }

}


template <typename T>
void run(const MatrixMap<T> *_input, MatrixMap<T> *_output)
{

    /*
    *  This kernel puts the values in a 4x4 block of Matrix A on the same row (Interleaved values)
    *         |a00 a01 a02 a03|
    *         |a10 a11 a12 a13|
    *         |a20 a21 a22 a23| = | a00 a10 a20 a30 || a01 a11 a21 a31 || a02 a12 a22 a32 || a03 a13 a23 a33 |
    *         |a30 a31 a32 a33|
    *
    *         After this operation, the output matrix will have the following shape: [ height * 4, ceil(width / 4.0f) ]
    */
    //gemm_interleave4x4(_input, _output);
    //(this->*_func)(_input, _output);
}