import numpy
import torch
import scipy
import matplotlib.pyplot as plt

import LibsControl

from wheels import *

dt = 0.001 


#three wheels connected with springs and controlled with two motors
#x state = (pos0, pos1, pos2, vel0, vel1, vel2)
#create dynamical system
ds = Wheels(dt)

print(ds)


#create loss weighting matrices (diagonal)
q = numpy.array([   [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] ] )
    
r = numpy.array( [ [0.1, 0.0], [0.0, 0.1] ]) 

#process and observation noise covariance
q_noise = 0.1*numpy.eye(ds.a.shape[0]) 
r_noise = 0.1*numpy.eye(ds.c.shape[0]) 


a_disc, b_disc, c_disc = LibsControl.c2d(ds.a, ds.b, ds.c, dt)

prediction_horizon = 8
#solve MPC controller
mpc = LibsControl.MPC(a_disc, b_disc, q, r, prediction_horizon)


 


#process simulation

n_max = 5000

#required output, 1 rad for all wheels
xr = numpy.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]).T

#observed state
x_hat = torch.zeros((ds.a.shape[0], 1), dtype=torch.float32)


#result log
t_result = []
u_result = []
x_result = []


#random initial state
x_initial = numpy.random.randn(ds.a.shape[0], 1)
ds.reset(x_initial)

#plant output
y = ds.y

for n in range(n_max):
    print("n = ", n)

    x = ds.x

    #compute controller output
    u = mpc.forward(xr, x)
    
    #compute plant output
    x, y = ds.forward_state(u)
  

    t_result.append(n*dt)
    u_result.append(u[:, 0].copy())
    x_result.append(x[:, 0].copy())
    
    
t_result = numpy.array(t_result)
x_result = numpy.array(x_result)
u_result = numpy.array(u_result)


#plot results
LibsControl.plot_closed_loop_response(t_result, u_result, x_result, x_hat = None, file_name = "mpc_output.png", u_labels = ["left", "right"], x_labels = ["position 0 [rad]", "position 1 [rad]", "position 2 [rad]", "velocity 0 [rad/s]", "velocity 1 [rad/s]", "velocity 2 [rad/s]"])





'''
#run demo

#random initial state
x_initial = numpy.random.randn(ds.a.shape[0], 1)
ds.reset(x_initial)

#required output, 1 rad for all wheels
yr = numpy.array([[1.0, 1.0, 1.0]]).T

#observed state
x_hat = numpy.zeros((ds.a.shape[0], 1))

#initial error integral
integral_action = numpy.zeros((ds.b.shape[1], 1))


#plant output
y = ds.y

disturbance = 0

n = 0

while True:
    #compute controller output
    u, integral_action, x_hat = mpc.forward(yr, y, integral_action, x_hat)
    
    u+= disturbance
    #compute plant output
    x, y = ds.forward_state(u)

    if n%10 == 0:
        ds.render()

    n+= 1
    if n%4000 == 0:
        disturbance = 50*(2.0*numpy.random.rand(2, 1) - 1.0)
'''