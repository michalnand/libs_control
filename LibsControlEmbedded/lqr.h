#ifndef _LQR_H_
#define _LQR_H_

#include <stdint.h>
#include <mmt.h>

template<unsigned int system_order, unsigned int inputs_count>
class LQR
{
    public:
        LQR()
        {
            mmt::fill<system_order*inputs_count, float>(this->k,  0);
            mmt::fill<system_order*inputs_count, float>(this->ki, 0);

            mmt::fill<system_order, float>(e_sum, 0);
            mmt::fill<inputs_count, float>(u, 0);
        } 

        void init(float *k, float *ki, float antiwindup)
        {
            mmt::copy<system_order*inputs_count, float>(this->k,  k);
            mmt::copy<system_order*inputs_count, float>(this->ki, ki);

            mmt::fill<system_order, float>(e_sum,   0);
            mmt::fill<inputs_count, float>(u,       0);

            this->antiwindup = antiwindup;
        }

        float* step(float *xr, float *x, float dt)
        {
            //compute error and integral action
            //error = xr - x
            //error_sum_new = error_sum + error*self.dt
            mmt::saxpy<inputs_count, float>(error_sum, xr, dt);
            mmt::saxpy<inputs_count, float>(error_sum, x, -dt);
                
            //LQR controll law
            //u = -self.k@x + self.ki@error_sum_new
            mmt::mm<inputs_count, system_order, 1, float>(u, k,  x, nullptr,  -1, 0);
            mmt::mm<inputs_count, system_order, 1, float>(u, ki, x, error_sum, 1, 1);
 
            //antiwindup
            //error_sum-= (u - u_sat)*dt
            antiwindup(dt);
            
            //return u_sat
            return u;
        } 

    private:
        void antiwindup(float dt)
        {
            for (unsigned int i = 0; i < inputs_count; i++)
            {
                float u_    = u[i]
                float u_sat = u_;

                if (u_sat < -antiwindup)
                {
                    u_sat = -antiwindup;
                }

                if (u_sat > antiwindup)
                {
                    u_sat = antiwindup;
                } 

                error_sum[i]-= (u_ - u_sat)*dt;

                u[i] = u_sat;
            }
        }

    private:
        float k[system_order*inputs_count];
        float ki[system_order*inputs_count];
        float e_sum[system_order];
        float u[inputs_count];

        float antiwindup;    
};

#endif