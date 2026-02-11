import numpy

import matplotlib.pyplot as plt

from reaction_wheel_inverted_pendulum   import *
from energy_lqr_controller              import *
from gain_scheduling_lqr_controller     import *

# unit step response, no feedback
def system_trajectory_response(ds, num_steps):
    n = ds.x.shape[0]

    x = numpy.zeros((n, 1))
    u = numpy.zeros((1, 1))

    x[0, 0] = numpy.pi
    x_result = []

    ds.reset()

    for n in range(num_steps):
        x = ds.step(x, u)
        x_result.append(x[:, 0])

    x_result = numpy.array(x_result)

    return x_result



# test controller
def system_stabilise(ds, controller, initial_theta, num_steps):
    n = ds.x.shape[0]

    x = numpy.zeros((n, 1))
    x[0, 0] = initial_theta
    
    u_result    = []
    x_result    = []
    alpha_result   = []

    ds.reset()

    for n in range(num_steps):

        xr = numpy.zeros((4, 1))
        #if n > num_steps//2:
        #    xr[0, 0] = numpy.pi/4
        
        u, a = controller.step(xr, x)

        x = ds.step(x, u)

        u_result.append(u[:, 0])
        x_result.append(x[:, 0])
        alpha_result.append(1.0*a)

    u_result    = numpy.array(u_result)
    x_result    = numpy.array(x_result)
    alpha_result= numpy.array(alpha_result)

    return u_result, x_result, alpha_result

if __name__ == "__main__":

    m   = 0.3
    l   = 0.1
    Ip  = 0.0007
    Iw  = 0.02
    dt  = 0.001

    num_steps = 1000

    time_steps = numpy.arange(num_steps)*dt


    ds_gt = ReactionWheelInvertedPendulum(m, l, Ip, Iw, dt=dt)
    x_result_gt = system_trajectory_response(ds_gt, num_steps)


    #controller = EnergyLQRController(m, l, Ip, Iw, dt=dt)
    controller = GainSchedulingLQRController(m, l, Ip, Iw, dt=dt)


    u_result, x_result, a_result = system_stabilise(ds_gt, controller, numpy.pi, num_steps)

    fig, axs = plt.subplots(3)
    #fig.suptitle('Vertically stacked subplots')
    axs[0].plot(time_steps, u_result[:, 0], color="red", lw=1.5)
    axs[0].set_xlabel("time [s]")
    axs[0].set_ylabel("torque")

    axs[1].plot(time_steps, x_result[:, 0]*180/numpy.pi, color="blue", lw=1.5)
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("angle [degrees]")

    axs[2].plot(time_steps, a_result, color="purple", lw=1.5)
    axs[2].set_xlabel("time [s]")
    axs[2].set_ylabel("gain scheduler")
    plt.show()

    print("done")