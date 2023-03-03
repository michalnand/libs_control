#ifndef _MATRIX_H_
#define _MATRIX_H_

'''
matmul,
Y = A@B + C

A : MxK
B : KxN
C : MxN
Y : MxN
'''
template<class DType>
void mm(DType *y, DType *a, DType *b, DType *c, unsigned int M, unsigned int N, unsigned int K, DType alpha = 1, DType beta = 1)
{
    for (unsigned int m = 0; m < M; m++)
        for (unsigned int n = 0; n < N; n++)
        {
            DType sum = 0;
            for (unsigned int k = 0; k < K; k++)
            {
                sum+= a[m*K + k]*b[k*N + n];
            }
        }

    if (c != nullptr)
    {
        for (unsigned int i = 0; i < M*N; i++)
        {
            y[i]+= c[i];
        }
    }
}



template<unsigned int M, unsigned int N, unsigned int K, class DType>
void mmt(DType *y, DType *a, DType *b, DType *c, DType alpha = 1, DType beta = 1)
{
    for (unsigned int m = 0; m < M; m++)
        for (unsigned int n = 0; n < N; n++)
        {
            DType sum = 0;
            for (unsigned int k = 0; k < K; k++)
            {
                sum+= a[m*K + k]*b[k*N + n];
            }

            y[m*N + n] = sum*alpha;
        }

    

    if (c != nullptr)
    {
        for (unsigned int i = 0; i < M*N; i++)
        {
            y[i]+= c[i]*beta;
        }
    }
}

#endif