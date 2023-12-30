#ifndef _LQR_H_
#ifndef _LQR_H_

#include <matmul.h>

constexpr void _select_max(unsigned int a, unsigned int b)
{
    if (a > b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

template<unsigned int inputs_count, unsigned int outputs_count, unsigned int order, class DType>
class LQG
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

            for (unsigned int i = 0; i < order*order; i++)
            {
                this->a[i] = 0;
            }

            for (unsigned int i = 0; i < order*inputs_count; i++)
            {
                this->b[i] = 0;
            }

            for (unsigned int i = 0; i < order*order; i++)
            {
                this->f[i] = 0;
            }

            for (unsigned int i = 0; i < order; i++)
            {
                this->x_hat[i] = 0;
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

        void set_a(DType value, unsigned int j, unsigned int i)
        {
            this->a[j*order + i] = value;
        }

        void set_b(DType value, unsigned int j, unsigned int i)
        {
            this->a[j*order + i] = value;
        }

        void set_f(DType value, unsigned int j, unsigned int i)
        {
            this->a[j*order + i] = value;
        }

        DType* forward(DType *xr, DType *y, DType dt)
        {
            //1, estimate x_hat using kalman observer

            //y_hat = Cx_hat
            //e = y - y_hat
            //dxhat = Ax_hat + Bu + F(y - y_hat)
            //x_hat = x_hat + dxhat*dt

            mmt<inputs_count, order, 1, DType>(this->y_hat, this->c, this->x_hat, nullptr);

            for (unsigned int i = 0; i < outputs_count; i++)
            {
                e[i] = y[i] - y_hat[i];
            }

            mmt<inputs_count, order, 1, DType>(this->x_hat, this->a, this->x_hat,   this->x_hat, dt);
            mmt<inputs_count, order, 1, DType>(this->x_hat, this->b, this->u,       this->x_hat, dt);
            mmt<inputs_count, order, 1, DType>(this->x_hat, this->f, this->e,       this->x_hat, dt);


            //2, use LQR with x_hat state 

            //compute scaling and error
            //e = xr*g - x
            for (unsigned int i = 0; i < inputs_count; i++)
            {
                this->e[i] = xr[i]*this->g[i] - x_hat[i];
            }

            //apply lQR control law
            //u = k*e
            mmt<inputs_count, order, 1, DType>(this->u, this->k, this->e, nullptr);

            return this->u;
        }

    private:
        //LQR matrices
        DType   k[inputs_count*order];
        DType   g[order];
        
        DType   u[inputs_count];
        DType   e[_select_max(inputs_count, order)];

        //Kalman observer matrices
        DType   a[order*order];
        DType   b[order*inputs_count];
        DType   f[order*order];
        
        DType   x_hat[order];
};

#endif
