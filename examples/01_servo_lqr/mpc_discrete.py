import numpy
import scipy
import matplotlib.pyplot as plt

import LibsControl


dt = 0.01 

# create dynamical system
# dx = Ax + Bu
# servo model, 3rd order
# state x = (position, angular velocity, current)

mat_a = numpy.zeros((3, 3))
mat_b = numpy.zeros((3, 1))

J = 0.02    #rotor moment of inertia (kg.m^2)
b = 0.2     #drag coefficient (N/m/s)
K = 0.3     #motor constant (N.m.A^-1)
R = 2.0     #wiring resitance, (ohm)
L = 0.4     #wiring inductance, (H)

mat_a[0][1] = 1.0
mat_a[0][0] = 0.0 #-2.3

mat_a[1][1] = -b/J
mat_a[1][2] = K/J

mat_a[2][1] = -K/J
mat_a[2][2] = -R/L

mat_b[2][0] = 1.0/L

mat_c = numpy.eye(mat_a.shape[0])


#create dynamical system
ds = LibsControl.DynamicalSystem(mat_a, mat_b, None, dt)

a_disc, b_disc, c_disc = LibsControl.c2d(mat_a, mat_b, mat_c, dt)

#ds = LibsControl.DynamicalSystemDiscrete(a_disc, b_disc, c_disc)

print(ds)


#create loss weighting matrices (diagonal)
q = numpy.array([ [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ] )
r = numpy.array( [ [10**-1] ]) 




#solve LQR controller
mpc = LibsControl.MPC(a_disc, b_disc, q, r, 128)



#process simulation

n_max = 1000

#required state, 100degrees
xr = numpy.zeros((mat_a.shape[0], 1))
xr[0][0] = 100.0*numpy.pi/180.0

u = numpy.zeros((mat_b.shape[1], 1))


#result log
t_result = []
u_result = []
x_result = []


#initial motor state
ds.reset()

for n in range(n_max):

    #plant state
    x = ds.x

    #compute controller output
    u = mpc.forward(xr, x, u)
    
    #compute plant output
    #x, y = ds.forward_state(u)
    x, y = ds.forward_state(u)
  
    #add constant disturbance in middle
    if n > n_max//2:
        x[0]+= 0.1*numpy.pi/180.0

    t_result.append(n*dt)
    u_result.append(u[:, 0].copy())
    x_result.append(x[:, 0].copy())

    
    
t_result = numpy.array(t_result)
x_result = numpy.array(x_result)
u_result = numpy.array(u_result)

#convert radians to degrees
x_result[:, 0]*= 180.0/numpy.pi
x_result[:, 1]*= 180.0/numpy.pi

#plot results
LibsControl.plot_closed_loop_response(t_result, u_result, x_result, x_hat = None, file_name = "mpc_discrete_output.png", u_labels = "voltage [V]", x_labels = ["position [deg]", "velocity [deg/s]", "current [A]"])
