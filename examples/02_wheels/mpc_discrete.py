import numpy
import torch
import scipy
import matplotlib.pyplot as plt

import LibsControl

from wheels import *

dt = 1.0/250.0


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
    
r = numpy.array( [ [10**-3, 0.0], [0.0, 10**-3] ]) 


a_disc, b_disc, c_disc = LibsControl.c2d(ds.a, ds.b, ds.c, dt)

control_horizon    = 16
prediction_horizon = 64
#solve MPC controller
mpc = LibsControl.MPC(a_disc, b_disc, q, r, control_horizon, prediction_horizon)


print(mpc.phi.shape) 
print(mpc.omega.shape) 
print(mpc.sigma.shape) 


#process simulation

n_max = 1000

#required output, 1 rad for all wheels
xr = numpy.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]).T
u = numpy.zeros((b_disc.shape[1], 1))

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
    x = ds.x

    #compute controller output
    u = mpc.forward(xr, x, u)
    
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






#run demo

#required output, 1 rad for all wheels
xr = numpy.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]).T
u = numpy.zeros((b_disc.shape[1], 1))


#random initial state
x_initial = numpy.random.randn(ds.a.shape[0], 1)
ds.reset(x_initial)



n = 0

while True:
    #compute controller output
    u = mpc.forward(xr, x, u)
    
    #compute plant output
    x, y = ds.forward_state(u)

    if n%10 == 0:
        ds.render()

    n+= 1
    if n%1000 == 0:
        xr = -xr
