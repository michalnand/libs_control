#ifndef _LQR_H_
#ifndef _LQR_H_

#include <matmul.h>

template<unsigned int inputs_count, unsigned int order, class DType>
class LQR
{
    public:
        LQR()
        {
            for (unsigned int i = 0; i < (inputs_count*order); i++)
            {
                this->k[i] = 0;
            }

            for (unsigned int i = 0; i < inputs_count; i++)
            {
                this->g[i] = 0;
            }

            for (unsigned int i = 0; i < inputs_count; i++)
            {
                this->u[i] = 0;
            }

            for (unsigned int i = 0; i < inputs_count; i++)
            {
                this->e[i] = 0;
            }
        }

        virtual ~LQR()
        {

        }

        void set_k(DType value, unsigned int j, unsigned int i)
        {
            this->k[j*order + i] = value;
        }

        void set_g(DType value, unsigned int j)
        {
            this->g[j] = value;
        }

        DType* forward(DType *xr, DType *x)
        {
            //compute scaling and error
            //e = xr*g - x
            for (unsigned int i = 0; i < inputs_count; i++)
            {
                this->e[i] = xr[i]*this->g[i] - x[i];
            }

            //apply lQR control law
            //u = k*e
            mmt<inputs_count, order, 1, DType>(this->u, this->k, this->e, nullptr);

            return this->u;
        }

    private:
        DType   k[inputs_count*order];
        DType   g[order];
        
        DType   u[inputs_count];
        DType   e[inputs_count];
};

#endif
