#include <iostream>
#include<cstdlib>
#include<ctime>
#include <time.h>

#include "header/Type.h"
#include "header/MatrixMap.h"
#include "header/NEGEMMInterleave4x4Kernel.h"

#include "NEGEMMInterleave4x4Kernel.cpp"
#include "MatrixMap.cpp"
#include "run_NEGEMM_u4.cpp"


//#define M 8*15
//#define K 4*15
//#define N 32*15

using namespace std;

int8_t getRandomNumber() { 
    int8_t min = 0; int8_t max = 7;
    static const double fraction = 1.0 / (RAND_MAX + 1.0); 
    return min + static_cast<int8_t>((max - min + 1) * (std::rand() * fraction)); 
}

void gemm_nn(int gM, int gN, int gK, float ALPHA,  int4_t *inA, int lda,int4_t *inB, int ldb,int16_t *C, int ldc)
{
    clock_t start, end;
    double result;
    start = clock();

    int i,j,k;
    for(i = 0; i < gM; ++i){
        for(j = 0; j < gN; ++j){
            for(k = 0; k < gK/2; ++k){
                int4_t A_PART = {inA[i*lda/2+k].val0, inA[i*lda/2+k].val1};
                if(j%2==0){
                    C[i*ldc+j] += A_PART.val0 * inB[k*ldb/2+j/2].val0;
                    C[i*ldc+j] += A_PART.val1 * inB[(k+1)*ldb/2+j/2].val0;

                }else{
                    C[i*ldc+j] += A_PART.val0*inB[k*ldb/2+j/2].val1;
                    C[i*ldc+j] += A_PART.val1*inB[(k+1)*ldb/2+j/2].val1;

                }

            }
        }
    }

    end = clock();
    result = (double)(end - start);
    printf("\n%f\n\n", result/CLOCKS_PER_SEC);

    //printf("\ngemm_NN elapsed time = %f (ms)\n\n", result);
}
int main()
{
    for(int i =15; i < 16; i++){

    int M = 8*i;
    int K = 4*i;
    int N = 32*i;    

    int4_t* input           = new int4_t[ M*K ];
    int4_t* weight          = new int4_t[ N*K ];
    int4_t* in_interleave   = new int4_t[ M*K ];
    int16_t* output         = new int16_t[ N*M ];


    for(int i = 0; i < M ;i ++){

        for(int j = 0; j < K/2 ; j++){

            (input + i * K/2 + j)->val0 =getRandomNumber();
            (input + i * K/2 + j)->val1 =getRandomNumber();

        }
    }
    for(int i = 0; i < K ;i ++){

        for(int j = 0; j < N/2 ; j++){
            

            (weight + i * N/2 + j)->val0 =getRandomNumber();
            (weight + i * N/2 + j)->val1 =getRandomNumber();

        }
    }

    MatrixMap<int4_t> mtx_in            = {input, M, K};
    MatrixMap<int4_t> mtx_we            = {weight, K, N};
    MatrixMap<int4_t> mtx_in_interleave = {in_interleave, 1, M * K };
    MatrixMap<int16_t> mtx_out          = {output, M, N };

    // for(int i =0 ;i < M; i++){
    //     for(int j =0 ; j < K/2; j++)

    //         printf("%d %d || ", mtx_in.ptr()[i*K/2+j].val0, mtx_in.ptr()[i*K/2+j].val1);

    //     cout << "\n";


    // }
    // cout << "\n"; 

    // for(int i = 0; i < K ;i ++){

    //     for(int j = 0; j < N/2 ; j++){

    //         printf("%d %d || ", mtx_we.ptr()[ i * N/2 + j].val0, mtx_we.ptr()[ i * N/2 + j].val1);


    //     }
    //     cout << "\n";

    // }
    gemm_nn( M,  N,  K,  1.0, input, K,weight, N,output, N);
    

    // cout << "\n=============run_matrix_matrix_multiply_s4 start==========================\n" << endl;
    clock_t start, end;
    double result;
    start = clock();
    run_matrix_matrix_multiply_s4(  &mtx_in,
                                    &mtx_we,
                                    &mtx_out,
                                    1.0 );
    end = clock();
    result = (double)(end - start);

    //printf("\nrun_matrix_matrix_multiply_s4 elapsed time = %f (ms)\n\n", result);
    printf("\n%f\n\n", result/CLOCKS_PER_SEC);
    printf("\n%ld\n\n", CLOCKS_PER_SEC);



    // for(int i =0 ; i < M; i++){

    //     for(int j=0; j < N; j++)
    //         printf("%4d ",mtx_out.ptr()[ i*N + j ]);
    //     cout << "\n";
    // }

    // cout << "\n=============run_matrix_matrix_multiply_s4 finish=========================\n" << endl ;
   }

    return 0;
}
