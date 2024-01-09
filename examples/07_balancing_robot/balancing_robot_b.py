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
        Jb= 0.005       # body inertia, along vertical axis (kg.m^2)
        g = 9.8;        # Gravity acceleration (m/s^2)
        l = 0.155;      # Half-length of pendulum (m)
        L = 0.08        # half length of wheel to wheel distance (m)


        self.mat_a = numpy.zeros((6, 6))
        self.mat_b = numpy.zeros((6, 2))
        self.mat_c = numpy.zeros((3, 6)) 

        den = J*(M + m) + M*m*(l**2) 


        self.mat_a[0][1] =  1.0
        self.mat_a[1][1] =  -(J + m*(l**2))*b/den
        self.mat_a[1][2] = (m**2)*g*(l**2)/den

        self.mat_a[2][3] =  1.0
        self.mat_a[3][1] = -m*b*l/den
        self.mat_a[3][2] = m*g*l*(M + m)/den


        self.mat_a[4][5] =  1.0
        #self.mat_a[5] = 0.1*numpy.random.randn(6)
        


        self.mat_b[1][0] = (J + m*(l**2))/den
        self.mat_b[3][0] = (m*l)/den
        self.mat_b[1][1] = (J + m*(l**2))/den
        self.mat_b[3][1] = (m*l)/den

        self.mat_b[5][0] =  L/Jb
        self.mat_b[5][1] = -L/Jb
    

        self.mat_c[0][0] = 1.0
        self.mat_c[1][2] = 1.0
        self.mat_c[2][4] = 1.0
        

        self.renderer = Render(700, 700)


    def forward(self, x, u):
        dx = self.mat_a@x + self.mat_b@u

        x  = x + dx*self.dt
        y  = self.mat_c@x
        
        return x, y

    
    def render(self, y):
        x_pos = y[0][0]
        y_pos = 0
        theta = y[1][0]*180.0/numpy.pi
        phi   = y[2][0]*180.0/numpy.pi

        self.renderer.render(x_pos, y_pos, phi, theta)