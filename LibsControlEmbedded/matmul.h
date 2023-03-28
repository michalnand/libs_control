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

            y[m*N + n] = alpha*sum;
        }

    if (c != nullptr)
    {
        for (unsigned int i = 0; i < M*N; i++)
        {
            y[i]+= beta*c[i];
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

            y[m*N + n] = alpha*sum;
        }

    

    if (c != nullptr)
    {
        for (unsigned int i = 0; i < M*N; i++)
        {
            y[i]+= beta*c[i];
        }
    }
}


template<class DType>
void fill(DType *x, unsigned int M, unsigned int N, DType value)
{
    for (unsigned int i = 0; i < M*N; i++)
    {
        x[i] = value;
    }
}

template<unsigned int M, unsigned int N, class DType>
void fill(DType *x, DType value)
{
    for (unsigned int i = 0; i < M*N; i++)
    {
        x[i] = value;
    }
}



template<class DType>
void copy(DType *dest, DType *src, unsigned int M, unsigned int N)
{
    for (unsigned int i = 0; i < M*N; i++)
    {
        dest[i] = src[i];
    }
}

template<unsigned int M, unsigned int N, class DType>
void copy(DType *dest, DType *src)
{
    for (unsigned int i = 0; i < M*N; i++)
    {
        dest[i] = src[i];
    }
}




template<class DType>
void add(DType *y, DType *a, DType *b, unsigned int M, unsigned int N)
{
    for (unsigned int i = 0; i < M*N; i++)
    {
        y[i] = a[i] + b[i];
    }
}

template<unsigned int M, unsigned int N, class DType>
void add(DType *y, DType *a, DType *b)
{
    for (unsigned int i = 0; i < M*N; i++)
    {
        y[i] = a[i] + b[i];
    }
}


template<class DType>
void sub(DType *y, DType *a, DType *b, unsigned int M, unsigned int N)
{
    for (unsigned int i = 0; i < M*N; i++)
    {
        y[i] = a[i] - b[i];
    }
}

template<unsigned int M, unsigned int N, class DType>
void sub(DType *y, DType *a, DType *b)
{
    for (unsigned int i = 0; i < M*N; i++)
    {
        y[i] = a[i] - b[i];
    }
}





template<class DType>
void mac(DType *y, DType *a, DType *b, DType alpha, DType beta, unsigned int M, unsigned int N)
{
    for (unsigned int i = 0; i < M*N; i++)
    {
        y[i] = y[i] + alpha*a[i] + beta*b[i];
    }
}

template<unsigned int M, unsigned int N, class DType>
void mac(DType *y, DType *a, DType *b, DType alpha, DType beta)
{
    for (unsigned int i = 0; i < M*N; i++)
    {
        y[i] = y[i] + alpha*a[i] + beta*b[i];
    }
}



#endif