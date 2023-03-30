import numpy
from render import *

'''
motor torque model : 

u = (Kt*Vm) / (Rm*r) = (Kt/r)*Im -> Im = u*r/Kt
Vm  : motor voltage    [V] 
Im  : motor current    [A]
Rm  : wiring resitance [ohm]
r   : wheel radius, [m]
Kt  :  motor torque constant, [N.m.A^-1], = torque/Im = 1/Kv[Si] = 60/(2*pi*Kv[RPM])
'''
class BalancingRobot:
    def __init__(self, dt = 0.001, velocity_model = False):

        self.dt = dt

        M = 0.5;        # Mass of carriage (kg)
        m = 0.95;       # Mass of pendulum (kg)
        b = 0.1;        # Drag Coefficient (N/m/s)
        J = 0.0076;     # Pendulum Moment of Inertia (kg.m^2)
        g = 9.8;        # Gravity acceleration (m/s^2)
        l = 0.155;      # Half-length of pendulum (m)


        self.mat_a = numpy.zeros((4, 4))
        self.mat_b = numpy.zeros((4, 1))
        self.mat_c = numpy.zeros((2, 4)) 

        den = J*(M + m) + M*m*(l**2) 


        self.mat_a[0][1] =  1.0
        self.mat_a[1][1] =  -(J + m*(l**2))*b/den
        self.mat_a[1][2] = (m**2)*g*(l**2)/den

        self.mat_a[2][3] =  1.0
        self.mat_a[3][1] = -m*b*l/den
        self.mat_a[3][2] = m*g*l*(M + m)/den


        self.mat_b[1][0] = (J + m*(l**2))/den
        self.mat_b[3][0] = (m*l)/den
    

        if velocity_model:
            self.mat_c[0][1] = 1.0
        else:
            self.mat_c[0][0] = 1.0

        self.mat_c[1][2] = 1.0
        

        self.renderer = Render(512, 512)


    def forward(self, x, u):
        dx = self.mat_a@x + self.mat_b@u

        x  = x + dx*self.dt
        y  = self.mat_c@x
        
        return x, y

    
    def render(self, y):
        x_pos = y[0][0]
        theta = y[1][0]*180.0/numpy.pi

        
        self.renderer.render(x_pos, 0, theta)