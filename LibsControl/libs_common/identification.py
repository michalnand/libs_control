import numpy
import scipy.linalg
import torch


'''
intentification of discrete dynamical system 

x(n+1) = A*x(n) + B*u(n)

x : observed system states, matrix (samples x order)
u : system controll input, matrix (samples x inputs_count)

returns : 
A : matrix shape (order x order)
B : matrix shape (inputs_count x order)
'''
def identification(u, x):
    order   = x.shape[1]

    #shift values
    u_now    = u[0:-1, :]    
    x_now    = x[0:-1, :]
    x_next   = x[1:, :]

    #concatenate into single matrix
    w        = numpy.hstack([x_now, u_now])

    #moore-penrose pseudoinverse
    w_inv   = numpy.linalg.pinv(w)

    #solve for AB matrices
    ab      = w_inv@x_next
    
    #split to A and B
    a = ab[0:order, :]
    b = ab[order:, :]

    return a, b



if __name__ == "__main__":


    '''
    order           = 1
    inputs_count    = 1

    a   = numpy.random.randn(order, order)
    b   = numpy.random.randn(inputs_count, order)
    

    a[0][0] = 0.93
    b[0][0] = -100.7
    '''


    order           = 2
    inputs_count    = 1

    a   = numpy.zeros((order, order))
    b   = numpy.zeros((inputs_count, order))
    

    a[0][0] = 1.0
    a[0][1] = 1.0
    a[1][0] = 0.0
    a[1][1] = 0.85

    b[0][0] = -2.3
    


    a   = numpy.round(a, 3)
    b   = numpy.round(b, 3)

    print(a)
    print("\n")
    print(b)
    print("\n\n\n")

    steps = 1000


    u = numpy.zeros((1, inputs_count))
    x = numpy.zeros((1, order))
    
    u_result = numpy.zeros((steps, inputs_count))
    x_result = numpy.zeros((steps, order))
    

    for i in range(steps):
        u = 2.0*numpy.random.rand(1, inputs_count) - 1.0

        u_result[i] = u[0]
        x_result[i] = x[0] + 0.1*numpy.random.randn(x.shape[1])

        x = x@a  + u@b


    a_hat, b_hat = identification(u_result, x_result)

    a_hat   = numpy.round(a_hat, 3)
    b_hat   = numpy.round(b_hat, 3)

    print(a_hat)
    print("\n")
    print(b_hat)
    print("\n\n\n")
    