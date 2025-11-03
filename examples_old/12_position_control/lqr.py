import numpy
import scipy
import matplotlib.pyplot as plt

import LibsControl


n_steps = 1000


dt              = 1.0/400.0

wheel_brace     = 78.0
wheel_diameter  = 28.0

tau_forward     = 0.1
tau_turn        = 0.1


mat_a = numpy.zeros((4, 4))
mat_b = numpy.zeros((4, 2))

mat_a[0][1] = 1.0
mat_a[1][1] = -1.0/tau_forward

mat_a[2][3] = 1.0
mat_a[3][3] = -1.0/tau_turn

mat_b[1][0] = (wheel_diameter)*tau_forward
mat_b[3][1] = (wheel_diameter/wheel_brace)*tau_turn

# create continuous dynamical system
ds = LibsControl.DynamicalSystem(mat_a, mat_b, None, dt)

print(mat_a)
print(mat_b)


# step response input
u        = numpy.zeros((mat_b.shape[1], 1))
u[0][0]  = 1.0
u[1][0]  = 1.0

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


x_result[:, 2] = x_result[:, 2]*180.0/numpy.pi
x_result[:, 3] = x_result[:, 3]*180.0/numpy.pi

LibsControl.plot_open_loop_response(t_result, x_result, "lqri_open_loop.png", ["distance [mm]", "velocity [mm/s]", "angle [degrees]", "angular rate [deg/s]"])


# discretise system
mat_a, mat_b, mat_c = LibsControl.c2d(ds.a, ds.b, ds.c, dt)


print("discretised system\n")
print(mat_a, "\n")
print(mat_b, "\n")

# LQR controller synthetis

#create loss weighting matrices (diagonal)
q = numpy.diag([1.0, 0.0, 1.0, 0.0])
r = numpy.diag([10**1, 0.1])


lqr = LibsControl.LQRDiscrete(mat_a, mat_b, q, r)

print("\n")
print("controller matrix")
print(lqr.k)
print(lqr.ki)




u = numpy.zeros((mat_b.shape[1], 1))
u_int = numpy.zeros((mat_b.shape[1], 1))

# required position, 100 degrees
xr = numpy.zeros((mat_a.shape[0], 1))
xr[0][0] = 100.0
xr[2][0] = 100*numpy.pi/180.0

# result log
t_result = []
u_result = []
x_result = []

# step response
# initial state
ds.reset()

for n in range(n_steps):

    x = ds.x
    u, u_int = lqr.forward(xr, x, u_int)

    x, y = ds.forward_state(u)

    t_result.append(n*dt)
    u_result.append(u[:, 0].copy())
    x_result.append(x[:, 0].copy())
    

t_result = numpy.array(t_result)
u_result = numpy.array(u_result)
x_result = numpy.array(x_result)

x_result[:, 2] = x_result[:, 2]*180.0/numpy.pi
x_result[:, 3] = x_result[:, 3]*180.0/numpy.pi

LibsControl.plot_closed_loop_response(t_result, u_result, x_result, file_name="lqri_closed_loop.png", x_labels=["distance [mm]", "velocity [mm/s]", "angle [degrees]", "angular rate [deg/s]"])
