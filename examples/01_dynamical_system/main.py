import numpy
import matplotlib.pyplot as plt

import LibsControl




if __name__ == "__main__":
    k  = 1
    dt = 0.01

    A = numpy.array([
        [0, k],
        [-k, 0]
    ])

    B = numpy.array([
        [0.0],
        [0.0]
    ])


    ds_a = LibsControl.DynamicalSystem(A, B, None, dt)
    ds_b = LibsControl.DynamicalSystem(A, B, None, dt)

    xa = numpy.zeros((2, 1))
    xa[0, 0] = 1

    xb = numpy.zeros((2, 1))
    xb[0, 0] = 1

    u = numpy.zeros((1, 1))

    xa_result = []
    xb_result = []
    for n in range(1000):
        xa, _ = ds_a.forward(xa, u, False)
        xb, _ = ds_b.forward(xb, u, True)

        xa_result.append(xa[:, 0].copy())
        xb_result.append(xb[:, 0].copy())

    xa_result = numpy.array(xa_result)
    xb_result = numpy.array(xb_result)

    fig, axs = plt.subplots(count, 1, figsize=(8, 2*count))

    plt.plot(xa_result[:, 0], xa_result[:, 1], label="Euler solver", color="royalblue")
    plt.plot(xb_result[:, 0], xb_result[:, 1], label="Runge-Kutta 4 solver", color="red")
    plt.xlabel("x_0")
    plt.ylabel("x_1")
    plt.legend()
    plt.show()