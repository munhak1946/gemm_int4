#include "header/MatrixMap.h"
#include "header/NEGEMM_u4.h"
#include "NEGEMM_u4.cpp"


template <typename T>
void run_matrix_matrix_multiply_s4(const MatrixMap<T> *ina, const MatrixMap<T> *inb, MatrixMap<int16_t> *out, float alpha){


    auto const_cast_ina                 = const_cast<MatrixMap<T> *>(ina);
    auto const_cast_ina_ptr             = const_cast_ina->ptr();
    auto const_cast_ina_ptr_initial     = const_cast_ina->ptr();

    auto const_cast_inb                 = const_cast<MatrixMap<T> *>(inb);
    auto const_cast_inb_ptr             = const_cast_inb->ptr();
    auto const_cast_inb_ptr_initial     = const_cast_inb->ptr();

    auto out_ptr             = out->ptr();
    auto out_ptr_initial     = out->ptr();


    const int ina_stride     = const_cast_ina->col / 2;
    const int ina_width      = const_cast_ina->col;
    const int ina_height     = const_cast_ina->row;

    const int inb_stride     = const_cast_inb->col / 2;
    const int inb_width      = const_cast_inb->col;
    const int inb_height     = const_cast_inb->row;

    const int out_stride     = out->col;
    const int out_width      = out->col;
    const int out_height     = out->row;

    const int block_height   = ina_height / BLK_A;
    const int partial_height = ina_height % BLK_A;

    const int block_width   = inb_width / BLK_B;
    partial_width = inb_width % BLK_B;
//    printf("inb_width = %d BLK_B = %d partial_width = %d\n", inb_width, BLK_B, partial_width);

    const int    num_elems_matrix_ina   = ina_width * ina_height /2;
    const int    num_elems_matrix_inb   = inb_width * inb_height /2;

    const_cast_ina_ptr_end_addr = const_cast_ina_ptr_initial + num_elems_matrix_ina;
    const_cast_inb_ptr_end_addr = const_cast_inb_ptr_initial + num_elems_matrix_inb;

    BLK_STATUS = true;
    for( int block = 0; block < block_width + (partial_width>0 ? 1:0); ++block ){
        // printf("hello world %d\n", block);

        const_cast_ina_ptr = const_cast_ina_ptr_initial;
        const_cast_inb_ptr = ( const_cast_inb_ptr_initial  += BLK_B / 2 * block);
        out_ptr            = ( out_ptr_initial             += BLK_B * block ) ;

        if(block == block_width)
            BLK_STATUS = false;
        int count = 0;
        for( ;const_cast_ina_ptr <= (const_cast_ina_ptr_end_addr - BLK_A * ina_stride ) ;){
            // printf("INA ROW main start inb block %d count = %d\n", block, count++);

            int4_t* ina_interleave                    = new int4_t[ BLK_A * ina_width ];
            const MatrixMap<int4_t> mtx_ina           = {const_cast_ina_ptr, BLK_A, ina_width};
            const MatrixMap<int4_t> mtx_inb           = {const_cast_inb_ptr, inb_height, inb_width};

            MatrixMap<int4_t> mtx_ina_interleave      = {ina_interleave, 1, BLK_A * ina_width };
            MatrixMap<int16_t> mtx_out                = {out_ptr, BLK_A, out_width};

            // cout << "\n=============interleave start  "<<block <<" ===================\n" << endl;

            gemm_interleave4x4(&mtx_ina, &mtx_ina_interleave);

            // for(int i =0 ;i < ina_width; i++){
            //     for(int j =0 ; j < BLK_A/2; j++)

            //         printf("%d %d || ", mtx_ina_interleave.ptr()[ i * BLK_A/2 + j ].val0, mtx_ina_interleave.ptr()[ i * BLK_A/2 + j ].val1);

            //     cout << "\n";

            // }
            // cout << "\n=============interleave finish  "<<block <<" ===================\n" << endl;

            // cout << "\n=============GEMM start==========================\n" << endl;

            matrix_matrix_multiply_s4(  &mtx_ina_interleave,
                                        &mtx_inb ,
                                        &mtx_out ,
                                        alpha);

            const_cast_ina_ptr  += BLK_A * ina_stride;
            out_ptr             += BLK_A * out_stride;
            // cout << "\n=============GEMM finish=========================\n" << endl ;

        }

        //INA ROW left over start
        for(; const_cast_ina_ptr < const_cast_ina_ptr_end_addr;){
        
            // printf("\nINA ROW left over start\n");
            int4_t* ina_left_over                     = new int4_t[ BLK_A * ina_width ];

            memcpy( ina_left_over,
                    const_cast_ina_ptr,
                    partial_height * ina_width);

            int4_t* ina_interleave                    = new int4_t[ BLK_A * ina_width ];
            const MatrixMap<int4_t> mtx_ina           = {ina_left_over, BLK_A, ina_width};
            MatrixMap<int4_t> mtx_ina_interleave      = {ina_interleave, 1, BLK_A * ina_width };
            MatrixMap<int16_t> mtx_out                = {out_ptr, partial_height, out_width};

            // cout << "\n=============interleave start INA ROW left over=====================\n" << endl;

            gemm_interleave4x4(&mtx_ina, &mtx_ina_interleave);

            // for(int i =0 ;i < ina_width; i++){
            //     for(int j =0 ; j < BLK_A/2; j++)
            //         printf("%d %d || ", mtx_ina_interleave.ptr()[ i * BLK_A/2 + j ].val0, mtx_ina_interleave.ptr()[ i * BLK_A/2 + j ].val1);
            //     cout << "\n";

            // }
            // cout << "\n=============interleave finish INA ROW left over===================\n" << endl;

            // cout << "\n=============GEMM start INA ROW left over==========================\n" << endl;

            matrix_matrix_multiply_s4(  &mtx_ina_interleave,
                                        inb,
                                        &mtx_out,
                                        alpha);

            const_cast_ina_ptr  += BLK_A * ina_stride;
            out_ptr             += BLK_A * out_stride;
            // cout << "\n=============GEMM finish INA ROW left over=========================\n" << endl ;


        }    //INA ROW left over end


    }



}