import numpy
import matplotlib.pyplot as plt

import LibsControl


dt = 0.01 

# create dynamical system
# dx = Ax + Bu
# servo model, 2nd order
# state x = (position, angular velocity)

mat_a = numpy.zeros((2, 2))
mat_b = numpy.zeros((2, 1))

tau = 0.7
k   = 17.0

mat_a[0][1] = 1.0
mat_a[1][1] = -1.0/tau

mat_b[1][0] = k/tau

#create dynamical system
ds = LibsControl.DynamicalSystem(mat_a, mat_b, None, dt)


print(ds)



a_disc, b_disc, c_disc = LibsControl.c2d(mat_a, mat_b, None, dt)



#create loss weighting matrices (diagonal)
q = numpy.array([ [1.0, 0.0], [0.0, 0.0], ] )
r = numpy.array( [ [0.01] ]) 

#solve LQR controller
lqr = LibsControl.LQRIDiscrete(a_disc, b_disc, q, r)


print("k  = ", lqr.k)
print("ki = ", lqr.ku)





#result log
t_result = []
u_result = []
x_result = []


#initial motor state
ds.reset()


n_max = 1000
x_req = numpy.zeros((n_max, mat_a.shape[0]))

x_req[:, 0] = 100.0*numpy.pi/180.0

#initial error integral
u = numpy.zeros((mat_b.shape[1], 1))


for n in range(n_max):

    #plant state
    x = ds.x

    #compute controller output
    xr = numpy.expand_dims(x_req[n, :], 1)
    u = lqr.forward(xr, x, u)
    
    #compute plant output
    x, y = ds.forward_state(u)
  
  

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
LibsControl.plot_closed_loop_response(t_result, u_result, x_result, x_hat = None, file_name = "lqr_output.png", u_labels = "voltage [V]", x_labels = ["position [deg]", "velocity [deg/s]", "current [A]"])


