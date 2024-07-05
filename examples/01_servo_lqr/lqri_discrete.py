import numpy
import scipy
import matplotlib.pyplot as plt

import LibsControl


def compute_jerk(x, dt):

    v   = (x[1:, :] - x[0:-1, :])/dt
    acc = (v[1:, :] - v[0:-1, :])/dt
    jerk= (acc[1:, :] - acc[0:-1, :])/dt

    return jerk

dt = 0.001 

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

mat_a[1][1] = -b/J
mat_a[1][2] = K/J

mat_a[2][1] = -K/J
mat_a[2][2] = -R/L

mat_b[2][0] = 1.0/L


#create dynamical system
ds = LibsControl.DynamicalSystem(mat_a, mat_b, None, dt)

print(ds)


#create loss weighting matrices (diagonal)
q = numpy.array([ [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ] )
r = numpy.array( [ [100.0] ]) 

a_disc, b_disc, c_disc = LibsControl.c2d(ds.a, ds.b, ds.c, dt)

print("discrete model")
print(a_disc)
print(b_disc)
print("\n\n")


#solve LQR controller
lqr = LibsControl.LQRIDiscrete(a_disc, b_disc, q, r)


print("k  = ", lqr.k)
print("ki = ", lqr.ki)


 

#process simulation

n_max = 10000

#required state, 100degrees
xr = numpy.zeros((mat_a.shape[0], 1))
xr[0][0] = 100.0*numpy.pi/180.0

#initial integral action
u = numpy.zeros((mat_b.shape[1], 1))

#result log
t_result = []
du_result = []
u_result = []
x_result = []


#initial motor state
ds.reset()

for n in range(n_max):

    #plant state
    x = ds.x

    #compute controller output
    u, du = lqr.forward(xr, x, u)
    
    #compute plant output
    x, y = ds.forward_state(u)
  
    #add constant distrubance in middle
    #if n > n_max//2:
    #    x[0]+= 0.1*numpy.pi/180.0

    t_result.append(n*dt)
    du_result.append(du[:, 0].copy())
    u_result.append(u[:, 0].copy())
    x_result.append(x[:, 0].copy())
    
    
t_result = numpy.array(t_result)
du_result = numpy.array(du_result)
u_result = numpy.array(u_result)
x_result = numpy.array(x_result)



#convert radians to degrees
x_result[:, 0]*= 180.0/numpy.pi
x_result[:, 1]*= 180.0/numpy.pi

#plot results
LibsControl.plot_closed_loop_response(t_result, u_result, x_result, x_hat = None, file_name = "lqri_discrete_output.png", u_labels = "voltage [V]", x_labels = ["position [deg]", "velocity [deg/s]", "current [A]"])



jerk_result = compute_jerk(x_result, dt)
print("total jerk = ", numpy.abs(jerk_result[:, 0]).sum())
