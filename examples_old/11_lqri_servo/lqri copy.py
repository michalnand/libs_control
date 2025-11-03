import numpy
import scipy
import matplotlib.pyplot as plt

import LibsControl


n_steps = 1000

dt = 0.001 


tau = 85*0.001
k   = 170.0

mat_a = numpy.zeros((2, 2))
mat_b = numpy.zeros((2, 1))

mat_a[0][0] = -20.0
mat_a[0][1] = 1.0
mat_a[1][1] = -1.0/tau

mat_b[0][0] = 0.0
mat_b[1][0] = k*1.0/tau


# create continuous dynamical system
ds = LibsControl.DynamicalSystem(mat_a, mat_b, None, dt)


# step response input
u        = numpy.zeros((mat_b.shape[1], 1))
u[0][0]  = 1.0

# result log
t_result = []
x_result = []

# step response
# initial state
ds.reset()

for n in range(n_steps):

    x, y = ds.forward_state(u)

    t_result.append(n*dt)
    x_result.append(x[:, 0].copy())
    

t_result = numpy.array(t_result)
x_result = numpy.array(x_result)


x_result[:, 0] = x_result[:, 0]*180.0/numpy.pi
x_result[:, 1] = x_result[:, 1]*60.0/(2.0*numpy.pi)

LibsControl.plot_open_loop_response(t_result, x_result, "lqri_open_loop.png", ["position [deg]", "velocity [rpm]"])


# discretise system
mat_a, mat_b, mat_c = LibsControl.c2d(ds.a, ds.b, ds.c, dt)


print("discretised system\n")
print(mat_a, "\n")
print(mat_b, "\n")

# LQR controller synthetis

#create loss weighting matrices (diagonal)
q = numpy.array([ [1.0, 0.0], [0.0, 0.0] ] )
#r = numpy.array( [ [10.0**3] ]) 

r = numpy.array( [ [10.0] ]) 

lqr = LibsControl.LQRIDiscrete(mat_a, mat_b, q, r)

print("\n")
print("controller matrix")
print(lqr.k)





u = numpy.zeros((mat_b.shape[1], 1))


# required position, 100 degrees
xr = numpy.zeros((mat_a.shape[0], 1))
xr[0][0] = 100*numpy.pi/180.0

# result log
t_result = []
u_result = []
x_result = []

# step response
# initial state
ds.reset()

for n in range(n_steps):

    x = ds.x
    u = lqr.forward(xr, x, u)

    x, y = ds.forward_state(u)

    t_result.append(n*dt)
    u_result.append(u[:, 0].copy())
    x_result.append(x[:, 0].copy())
    

t_result = numpy.array(t_result)
u_result = numpy.array(u_result)
x_result = numpy.array(x_result)

x_result[:, 0] = x_result[:, 0]*180.0/numpy.pi
x_result[:, 1] = x_result[:, 1]*60.0/(2.0*numpy.pi)

LibsControl.plot_closed_loop_response(t_result, u_result, x_result, file_name="lqri_closed_loop.png", x_labels=["position [deg]", "velocity [rpm]"])
