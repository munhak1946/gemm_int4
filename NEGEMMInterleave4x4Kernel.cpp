#include "NEGEMMInterleave4x4Kernel.h"

#include "header/MatrixMap.h"
#include "MatrixMap.cpp"

using namespce std;

void NEGEMMInterleave4x4Kernel::gemm_interleave4x4(const MatrixMap<float>* in, MatrixMap<float>* out)
{

    const int in_height = in->row;
    const int in_stride = 4;

    //const size_t partial_y = in_height % 4;

    int window_start_x = 0;
    int window_end_x = window_start_x + 4;

    //if(id.y() + 4 <= in_height)
    //{
        for(int x = window_start_x; x < window_end_x; ++x)
        {
            const ScalarType data[4] =
                    {
                            *(reinterpret_cast<const ScalarType *>(in->ptr() + 0 * in_stride) + x),
                            *(reinterpret_cast<const ScalarType *>(in->ptr() + 1 * in_stride) + x),
                            *(reinterpret_cast<const ScalarType *>(in->ptr() + 2 * in_stride) + x),
                            *(reinterpret_cast<const ScalarType *>(in->ptr() + 3 * in_stride) + x),
                    };
            std::memcpy(out->ptr() + x * 4 * sizeof(ScalarType), data, 4 * sizeof(ScalarType));
        }
    //}
/*    else
    {
        for(size_t x = window_start_x; x < window_end_x; ++x)
        {
            ScalarType data[4] = { 0, 0, 0, 0 };

            for(size_t y = 0; y < partial_y; ++y)
            {
                data[y] = *(reinterpret_cast<const ScalarType *>(in.ptr() + y * in_stride) + x);
            }

            std::memcpy(out.ptr() + x * 4 * sizeof(ScalarType), data, 4 * sizeof(ScalarType));
        }
    }
*/
}

void NEGEMMInterleave4x4Kernel::run(const MatrixMap *_input, MatrixMap *_output)
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
    (this->*_func)(_input, _output);
}