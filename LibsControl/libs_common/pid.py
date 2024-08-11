import numpy


'''
    dicrete PID controller : 
    u(n+1) = u(n) + k0e(n) + k1e(n-1) + k2e(n-2)
''' 
class PID:

    def __init__(self, kp, ki, kd, antiwindup = 10**10, du_max=10**10):
        self.k0 = kp + ki + kd
        self.k1 = -kp -2.0*kd
        self.k2 = kd

        self.e0 = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        
        
        self.antiwindup = antiwindup
        self.du_max     = du_max

    '''
    inputs:
        xr     : required output, float
        x      : system output, float
        u_prev : storage for integral action, float
        
    returns:
        u : input into plant, float
    '''
    def forward(self, xr, x, u_prev):
        # error compute
        self.e2 = self.e1
        self.e1 = self.e0
        self.e0 = xr - x

        du = self.k0*self.e0 + self.k1*self.e1 + self.k2*self.e2

        #kick clipping, maximum output value change limitation
        du  = numpy.clip(du, -self.du_max , self.du_max)

        #antiwindup, maximum output value limitation
        u   = numpy.clip(u_prev + du,  -self.antiwindup, self.antiwindup)

        return u

    def reset(self):
        self.e0 = 0.0
        self.e1 = 0.0
        self.e2 = 0.0