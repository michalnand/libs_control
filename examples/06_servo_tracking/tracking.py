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
mat_c = numpy.zeros((3, 3))


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


ds = LibsControl.DynamicalSystem(mat_a, mat_b, None, dt)

#discretise
a_disc, b_disc, c_disc = LibsControl.c2d(ds.a, ds.b, ds.c, dt)

print("discrete model")
print(a_disc)
print(b_disc)
print("\n\n")

#create loss weighting matrices (diagonal)
q = numpy.array([ [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ] )
r = numpy.array( [ [0.1] ]) 

#solve LQR controller
lqr = LibsControl.LQRIDiscrete(a_disc, b_disc, q, r)


#create loss weighting matrices (diagonal)
q = numpy.array([ [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0] ] )
r = numpy.array( [ [0.01] ]) 

#solve MPC controller
control_horizon    = 8
prediction_horizon = 32 
mpc = LibsControl.MPC(a_disc, b_disc, q, r, control_horizon, prediction_horizon)





#process simulation
n_max = 1000

xr_trajectory = numpy.zeros((n_max, mat_a.shape[0], 1))

#required states trajectory, from 0 to 100degrees in middle
for i in range(n_max//3):
    xr_trajectory[i + n_max//3, 0, 0] = 100.0*numpy.pi/180.0



#initial integral action
u = numpy.zeros((mat_b.shape[1], 1))

#result log
lqr_t_result = []
lqr_u_result = []
lqr_x_result = []


#initial motor state
ds.reset()

for n in range(n_max):

    #plant state
    x = ds.x

    #compute controller output
    u = lqr.forward(xr_trajectory[n], x, u)
    
    #compute plant output   
    x, y = ds.forward_state(u)
  
   
    lqr_t_result.append(n*dt)
    lqr_u_result.append(u[:, 0].copy())
    lqr_x_result.append(x[:, 0].copy())
    
    
lqr_t_result = numpy.array(lqr_t_result)
lqr_x_result = numpy.array(lqr_x_result)
lqr_u_result = numpy.array(lqr_u_result)

lqr_energy = (lqr_u_result**2).mean()
lqr_cost   = ((xr_trajectory[:, 0, 0] - lqr_x_result[:, 0])**2).mean()
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

for n in range(n_max - prediction_horizon):

    #plant state
    x = ds.x

    #compute controller output
    u = mpc.forward_trajectory(xr_trajectory[n:n+prediction_horizon], x, u)
    
    #compute plant output   
    x, y = ds.forward_state(u)
    
   
    mpc_t_result.append(n*dt)
    mpc_u_result.append(u[:, 0].copy())
    mpc_x_result.append(x[:, 0].copy())
    
    
mpc_t_result = numpy.array(mpc_t_result)
mpc_x_result = numpy.array(mpc_x_result)
mpc_u_result = numpy.array(mpc_u_result)

mpc_energy = (mpc_u_result**2).mean()
mpc_cost   = ((xr_trajectory[0:mpc_x_result.shape[0], 0, 0] - mpc_x_result[:, 0])**2)
mpc_cost   = mpc_cost.mean()

print("mpc_energy = ", mpc_energy)
print("mpc_cost   = ", mpc_cost)
print("\n\n")


plt.plot(lqr_t_result, xr_trajectory[:, 0, 0]*180.0/numpy.pi, label="required trajectory", color= "red")
plt.plot(lqr_t_result, lqr_x_result[:, 0]*180.0/numpy.pi, label="LQR optimal control", color = "deepskyblue")
plt.plot(mpc_t_result, mpc_x_result[:, 0]*180.0/numpy.pi, label="MPC control", color= "purple")
plt.legend()
plt.xlabel("time step [s]")
plt.ylabel("angle [degrees]")
plt.show()

#plot results
#LibsControl.plot_closed_loop_response(t_result, u_result, x_result, x_hat = None, file_name = "lqri_discrete_output.png", u_labels = "voltage [V]", x_labels = ["position [deg]", "velocity [deg/s]", "current [A]"])
