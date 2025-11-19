import numpy
import matplotlib.pyplot as plt

import LibsControl

def simulate(k, tau, steps, lw):
    A = numpy.array([
            [-1.0/tau],
        ])

    B = numpy.array([
        [k*1.0/tau],
    ])


    ds = LibsControl.DynamicalSystem(A, B, None, dt)

    u = numpy.zeros((1, 1))
    x = numpy.zeros((1, 1))

    t_result = []
    u_result = []
    x_result = []
    for n in range(steps):
        u[:, 0] = 1.0
        x, _ = ds.forward(x, u)

        # log results
        t_result.append(n*dt)
        u_result.append(u[:, 0])
        x_result.append(x[:, 0])

    t_result = numpy.array(t_result)
    u_result = numpy.array(u_result)
    x_result = numpy.array(x_result)    

    

    plt.plot(t_result, x_result, label="k="+str(k) + " , " + "tau=" + str(tau), lw=lw)

if __name__ == "__main__":
    dt  = 0.01

    
    k_values   = [0.5, 1.0, 1.5]
    tau_values = [0.1, 0.5, 1.0]
    
    plt.cla()


    
    n_steps = 1000

    for k in k_values:
        tau = 1.0
        simulate(k, tau, n_steps, 2)

    for tau in tau_values:
        k = 2
        simulate(k, tau, n_steps, 2)


    plt.xlabel("time [s]")
    plt.ylabel("output")
    plt.legend()
    plt.show()
