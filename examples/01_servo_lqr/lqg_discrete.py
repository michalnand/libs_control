import numpy
import scipy
import matplotlib.pyplot as plt

import LibsControl


dt = 0.001 

# create dynamical system
# dx = Ax + Bu
# servo model, 3rd order
# state x = (position, angular velocity, current)

mat_a = numpy.zeros((3, 3))
mat_b = numpy.zeros((3, 1))
mat_c = numpy.zeros((1, 3))


J = 0.02    #rotor moment of inertia (kg.m^2)
b = 0.2     #drag coefficient (N/m/s)
K = 0.3     #motor constant (N.m.A^-1)
R = 2.0     #wiring resitance, (ohm)
L = 0.4     #wiring inductance, (H)

mat_a[0][1] = 1.0

mat_a[1][1] = -b/J
mat_a[1][2] = K/J

mat_a[2][1] = -K/J
mat_a[2][2] = -R/L

mat_b[2][0] = 1.0/L

mat_c[0][0] = 1.0



#create dynamical system
ds = LibsControl.DynamicalSystem(mat_a, mat_b, mat_c, dt)

print(ds)


#create loss weighting matrices (diagonal)
q = numpy.array([ [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ] )
r = numpy.array( [ [200.0] ]) 

#process and observation noise covariance
q_noise = 0.001*numpy.eye(ds.a.shape[0]) 
r_noise = 0.1*numpy.eye(ds.c.shape[0]) 


a_disc, b_disc, c_disc = LibsControl.c2d(ds.a, ds.b, ds.c, dt)

#solve LQG controller
lqg = LibsControl.LQGDiscrete(a_disc, b_disc, c_disc, q, r, q_noise, r_noise, 10**10, 0.05)


print("k  = ", lqg.k)
print("ki = ", lqg.ki)
print("f  = ", lqg.f)




#process simulation

n_max = 10000

#required output, 100degrees
yr = numpy.zeros((mat_c.shape[0], 1))
yr[0][0] = 100.0*numpy.pi/180.0



#observed state
x_hat = numpy.zeros((mat_a.shape[0], 1))



#initial error integral
integral_action = numpy.zeros((mat_b.shape[1], 1))

#result log
t_result = []
u_result = []
x_result = []


#initial motor state
ds.reset()

#plant output
y = ds.y

for n in range(n_max):

    #compute controller output
    u, integral_action, x_hat = lqg.forward(yr, y, integral_action, x_hat)
    
    #compute plant output
    x, y = ds.forward_state(u)
  
    #add constant disturbance in middle
    if n > n_max//2:
        x[0]+= 0.5*numpy.pi/180.0

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
LibsControl.plot_closed_loop_response(t_result, u_result, x_result, x_hat = None, file_name = "lqg_discrete_output.png", u_labels = "voltage [V]", x_labels = ["position [deg]", "velocity [deg/s]", "current [A]"])
