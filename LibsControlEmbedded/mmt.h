#ifndef _MATRIX_H_
#define _MATRIX_H_


namespace mmt
{
    template<unsigned int N, class DType>
    void fill(DType *x, DType value)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            x[i] = value;
        }
    }

    template<unsigned int N, class DType>
    void copy(DType *y, DType *x)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            y[i] = x[i];
        }
    } 

    template<unsigned int N, class DType>
    void clip(DType *y, DType *x, DType min_value, DType max_value)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            DType v = x[i];
            if (v < min_value)
            {
                v = min_value;
            }
            else if (v > max_value)
            {
                v = max_value;
            }

            y[i] = v;
        }
    }


    template<unsigned int N, class DType>
    void sgn(DType *y, DType *x)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            DType v = x[i];

            if (v < 0)
            {
                v = -1;
            }
            else if (v > 0)
            {
                v = 1;
            }
            else
            {
                v = 0;
            }

            y[i] = v;
        }
    }


    template<unsigned int N, class DType>
    void abs(DType *y, DType *x)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            DType v = x[i];
            
            if (v < 0)
            {
                v = -v;
            }
            
            y[i] = v;
        }
    }




    template<unsigned int N, class DType>
    void add(DType *y, DType *a, DType *b)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            y[i] = a[i] + b[i];
        }
    }


    template<unsigned int N, class DType>
    void sub(DType *y, DType *a, DType *b)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            y[i] = a[i] - b[i];
        }
    }


    template<unsigned int N, class DType>
    void mac(DType *y, DType *a, DType *b, DType alpha)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            y[i] = y[i] + alpha*a[i]*b[i];
        }
    }



    template<unsigned int N, class DType>
    void saxpy(DType *y, DType *x, DType alpha)
    {
        for (unsigned int i = 0; i < N; i++)
        {
            y[i] = y[i] + alpha*a[i];
        }
    }


    '''
    matmul,
    Y = alpha*A@B + beta*C

    A : MxK
    B : KxN
    C : MxN
    Y : MxN
    '''
    template<unsigned int M, unsigned int N, unsigned int K, class DType>
    void mm(DType *y, DType *a, DType *b, DType *c, DType alpha = 1, DType beta = 1)
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


    template<unsigned int M, unsigned int N, class DType>
    void transpose(DType *y, DType *x)
    {
        for (unsigned int m = 0; m < M; m++)
            for (unsigned int n = 0; n < N; n++)
            {
                DType v = x[m*N + n];
                y[n*M + m] = v;
            }
    }



    //2x2 matrix inversion
    template<unsigned int M, unsigned int N, class DType>
    int mm_inv_2x2(DType *y, DType *x) 
    {
        float det = x[0*N + 0] * x[1*N + 1] - x[0*N + 1] * x[1*N + 0];

        if (det != 0) 
        {
            float inv_det = 1.0/det;

            y[0*N + 0] = x[1*N + 1] * inv_det;
            y[0*N + 1] = -x[0*N + 1] * inv_det;
            y[1*N + 0] = -x[1*N + 0] * inv_det;
            y[1*N + 1] = x[0*N + 0] * inv_det;

            return 0;
        } 
        else 
        {
           return -1;
        }
    }
    
}

#endif