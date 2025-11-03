import numpy
import matplotlib.pyplot as plt

import LibsControl

def ds_get_response(ds, x_initial, u_seq):
    x = x_initial.copy()

    n_steps = u_seq.shape[0]

    x_seq = []
    for n in range(n_steps):
        u    = numpy.expand_dims(u_seq[n, :], 1)
        x, _ = ds.forward(x, u)
        
        x_seq.append(x[:, 0])
    
    x_seq = numpy.array(x_seq) 

    return x_seq


if __name__ == "__main__":
    dt = 1.0/250.0

    mat_a = numpy.zeros((2, 2), dtype=numpy.float32)
    mat_b = numpy.zeros((2, 1), dtype=numpy.float32)
    mat_c = numpy.eye(2, dtype=numpy.float32)

    mat_a[0][1] = 1.0
    mat_a[1][0] = -300.0
    mat_a[1][1] = -2.0


    ds_ref = LibsControl.DynamicalSystem(mat_a, mat_b, mat_c, dt)
    print(ds_ref)

    
    x_initial       = numpy.zeros((2, 1), dtype=numpy.float32)
    x_initial[0][0] = 1.0

    u               = numpy.zeros((ds_ref.b.shape[1], 1))

    u_seq = numpy.zeros((1000, 1), dtype=numpy.float32)


    x_ref = ds_get_response(ds_ref, x_initial, u_seq)

    x_obs = x_ref + 0.001*numpy.random.randn(x_ref.shape[0], x_ref.shape[1])
    

    a_model, b_model = LibsControl.ls_identification(u_seq, x_obs)
    ds_model = LibsControl.DynamicalSystemDiscrete(a_model, b_model, mat_c)
    x_ls = ds_get_response(ds_model, x_initial, u_seq)


    a_model, b_model = LibsControl.krls_identification(u_seq, x_obs)
    ds_model = LibsControl.DynamicalSystemDiscrete(a_model, b_model, mat_c)
    x_krls = ds_get_response(ds_model, x_initial, u_seq)





    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    plt.plot(x_ref[:, 0], x_ref[:, 1], label="reference", linewidth=2, color="red")
    plt.plot(x_ls[:, 0], x_ls[:, 1], label="least squares", linewidth=1, color="blue")
    plt.plot(x_krls[:, 0], x_krls[:, 1], label="kalman em squares", linewidth=1, color="purple")

    plt.legend()
    plt.show()
