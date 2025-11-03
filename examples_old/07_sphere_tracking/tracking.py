import numpy
import scipy
import matplotlib.pyplot as plt

import LibsControl

dt = 1.0/100.0

max_u = 0.5

mat_a = numpy.zeros((4, 4))
mat_b = numpy.zeros((4, 2))

mat_a[0][1] = 1.0
mat_a[2][3] = 1.0

mat_b[1][0] = 1.0
mat_b[3][1] = 1.0

mat_c = numpy.eye(mat_a.shape[0])


#mat_a = numpy.random.randn(4, 4)
#mat_b = numpy.random.randn(4, 2)


#create dynamical system
ds = LibsControl.DynamicalSystem(mat_a, mat_b, None, dt)

print(ds)

a_disc, b_disc, c_disc = LibsControl.c2d(mat_a, mat_b, mat_c, dt)



print("discrete model")
print(a_disc)
print(b_disc)
print("\n\n")

#create loss weighting matrices (diagonal)
q = numpy.array([ [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]] )
r = numpy.array( [ [10**-3, 0.0], [0.0, 10**-3] ]) 

#solve LQR controller
lqr = LibsControl.LQRDiscrete(a_disc, b_disc, q, r, max_u)

print("k = ")
print(lqr.k)
print("ki = ")
print(lqr.ki)


#solve MPC controller
control_horizon    = 8
prediction_horizon = 64
q = numpy.array([ [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]] )
r = numpy.array( [ [10**-3, 0.0], [0.0, 10**-3] ]) 

mpc = LibsControl.MPC(a_disc, b_disc, q, r, control_horizon, prediction_horizon, max_u)

#mpc = LibsControl.MPPI(a_disc, b_disc, q, r, prediction_horizon, max_u)





#process simulation
path_length = 500


# create required trajectory - square motion pattern
xr_trajectory = []


for j in range(4):
    for i in range(path_length):
        x = [2.0*i/path_length - 1.0, 0, -1.0, 0]
        xr_trajectory.append(x)

    for i in range(path_length):
        x = [1.0, 0, 2.0*i/path_length - 1.0, 0]
        xr_trajectory.append(x)

    for i in range(path_length):
        x = [-(2.0*i/path_length - 1.0), 0, 1.0, 0]
        xr_trajectory.append(x)

    for i in range(path_length):
        x = [-1.0, 0, -(2.0*i/path_length - 1.0), 0]
        xr_trajectory.append(x)


xr_trajectory = numpy.array(xr_trajectory)

n_max = xr_trajectory.shape[0]


#initial integral action
integral_action = numpy.zeros((mat_b.shape[1], 1))

#result log
lqr_t_result = []
lqr_u_result = []
lqr_x_result = []


#initial motor state
ds.reset()

ds.x[0, 0] = xr_trajectory[0, 0]
ds.x[2, 0] = xr_trajectory[0, 2]

for n in range(n_max):
    #plant state
    x = ds.x

    #compute controller output
    xr = numpy.expand_dims(xr_trajectory[n], 1)
    u, integral_action = lqr.forward(xr, x, integral_action)

    #compute plant output   
    x, y = ds.forward_state(u)
  
    lqr_t_result.append(n*dt)
    lqr_u_result.append(u[:, 0].copy())
    lqr_x_result.append(x[:, 0].copy())
    
    
lqr_t_result = numpy.array(lqr_t_result)
lqr_x_result = numpy.array(lqr_x_result)
lqr_u_result = numpy.array(lqr_u_result)




lqr_energy = (lqr_u_result**2).mean()
lqr_cost   = ((xr_trajectory - lqr_x_result)**2).mean()
print("lqr_energy = ", lqr_energy)
print("lqr_cost   = ", lqr_cost)
print("\n\n")


#initial integral action
u = numpy.zeros((mat_b.shape[1], 1))

#result log
mpc_t_result = []
mpc_u_result = []
mpc_x_result = []


#initial motor state
ds.reset()

ds.x[0, 0] = xr_trajectory[0, 0]
ds.x[2, 0] = xr_trajectory[0, 2]


#for n in range(n_max - prediction_horizon):
for n in range(n_max - prediction_horizon):

    #plant state
    x = ds.x

    #compute controller output
    xr = xr_trajectory[n:n+prediction_horizon, :]
    u = mpc.forward_trajectory(xr, x, u)
    
    #compute plant output   
    x, y = ds.forward_state(u)
    
   
    mpc_t_result.append(n*dt)
    mpc_u_result.append(u[:, 0].copy())
    mpc_x_result.append(x[:, 0].copy())
    
    
mpc_t_result = numpy.array(mpc_t_result)
mpc_x_result = numpy.array(mpc_x_result)
mpc_u_result = numpy.array(mpc_u_result)


mpc_energy = (mpc_u_result**2).mean()
print("mpc_energy = ", mpc_energy)

mpc_cost   = ((xr_trajectory[0:n_max - prediction_horizon, :] - mpc_x_result)**2).mean()
print("mpc_cost   = ", mpc_cost)
print("\n\n")


plt.plot(xr_trajectory[:, 0], xr_trajectory[:, 2], label="required trajectory", color= "red", lw=4)
plt.plot(lqr_x_result[:, 0], lqr_x_result[:, 2], label="LQR trajectory", color= "lime", lw=2)
plt.plot(mpc_x_result[:, 0], mpc_x_result[:, 2], label="MPC trajectory", color= "deepskyblue", lw=2)

plt.legend()
plt.xlabel("position [m]")
plt.ylabel("position [m]")
plt.show()

