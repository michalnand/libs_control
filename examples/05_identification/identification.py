import numpy
import scipy
import matplotlib.pyplot as plt

import LibsControl

from wheels import *


def identification_test(u_seq, x_gt, x_input, identification_func):

    a_model, b_model = identification_func(u_seq, x_input)

    x_pred       = numpy.zeros(x_input.shape)
    x_pred[0, :] = x_input[0, :]
    for n in range(x_pred.shape[0]-1):
        x_pred[n + 1] = a_model@x_pred[n] + b_model@u_seq[n]

    error      = numpy.abs(x_gt - x_pred)
    error_mse  = (error**2).mean()
    error_mape = (error/(numpy.abs(x_gt) + 10**-10)).mean()

    error_seq = error.sum(axis=1)

    
    print("\n\n")
    print(numpy.round(a_model, 5))
    print(numpy.round(b_model, 5))
    print("\n\n")
    
    

    return round(error_mse, 4), round(error_mape, 4), error_seq




dt = 1.0/250.0


#three wheels connected with springs and controlled with two motors
#x state = (pos0, pos1, pos2, vel0, vel1, vel2)
#create dynamical system
ds = Wheels(dt)

u  = numpy.zeros((ds.b.shape[1], 1))


x_seq = []
u_seq = []

for n in range(1000):
    x, _ = ds.forward_state(u)

    #ds.render()
    if n%1 == 0:
        u = 0.1*numpy.random.randint(0, 3, (ds.b.shape[1], 1)) - 1
    
    u_seq.append(u[:, 0])
    x_seq.append(x[:, 0])



u_seq = numpy.array(u_seq) 
x_seq = numpy.array(x_seq) 



levels = numpy.abs(x_seq).mean(axis=0)

a_ref, b_ref, _ = LibsControl.c2d(ds.a, ds.b, ds.c, dt)

print(numpy.round(a_ref, 5))
print(numpy.round(b_ref, 5))


#noise level percent
#noise_levels = [0.0, 0.001, 0.01, 0.1]
noise_levels = [0.01]

for noise_level in noise_levels:
    x_noise  = noise_level*numpy.random.randn(x_seq.shape[0], x_seq.shape[1])
    x_noised = x_seq + x_noise

    error_ls, mape_ls, error_seq_ls   = identification_test(u_seq, x_seq, x_noised, LibsControl.ls_identification)
    error_rls, mape_rls, error_seq_rls = identification_test(u_seq, x_seq, x_noised, LibsControl.rls_identification)
    error_krls, mape_krls, error_seq_krls  = identification_test(u_seq, x_seq, x_noised, LibsControl.krls_identification)

    print(noise_level, error_ls, mape_ls, error_rls, mape_rls, error_krls, mape_krls)

    plt.xlabel("time step")
    plt.ylabel("MSE")
    plt.plot(error_seq_ls, label="least squares", linewidth=4, color="red")
    plt.plot(error_seq_rls, label="recursive least squares", color="lime")
    plt.plot(error_seq_krls, label="EM kalman recursive least squares", color="blue")
    
    plt.legend()
    plt.show()