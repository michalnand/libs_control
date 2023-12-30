import numpy
from .differences    import *
from .sparse_solver  import *
from .non_linear_lib import *

def identification(u, x, dt, steps_count=20, augmentations = [], higher_order_difference = False):

    if higher_order_difference:
        u_now = u[0:-6, :]
        x_now = x[0:-6, :]
        dx    = first_difference_6(x, dt)
    else:
        u_now = u[0:-1, :]
        x_now = x[0:-1, :]
        dx    = first_difference_1(x, dt)
    

    #concatenate into single matrix
    w   = numpy.hstack([x_now, u_now])

    #add augmented matrices
    w_aug = w.copy()
    for i in range(len(augmentations)):
        w_aug_tmp   = augmentations[i](w)
        w_aug       = numpy.hstack([w_aug, w_aug_tmp])
    
    solver             = SparseSolver()
    _, thetas, loss    = solver.solve(dx, w_aug, steps_count)

    return thetas, loss





'''
special module for servo plants identification, with inertia
model : 
dv = a*v + b*u
dx = v

model parameters : a, b

where : 
a = -1.0/tau
b = k/tau

tau : is time constant
k   : is amplification

u : control input,  shape (steps, 1)
x : servo position, shape (steps, 1)
'''
def servo_identification(u, x, dt):

    '''
    u    = u[2:, :]
    v    = first_difference_1(x, dt)[1:, :]
    acc  = second_difference_1(x, dt)
    


    w   = numpy.hstack([v, u])

    #solve model
    theta = numpy.linalg.lstsq(w, acc, rcond=None)[0]

    a = theta[0][0]
    b = theta[1][0]
    '''


    u    = u[2:, :]
    v    = first_difference_1(x, dt)[1:, :]
    


    w   = numpy.hstack([v, u])

    #solve model
    theta = numpy.linalg.lstsq(w, acc, rcond=None)[0]

    a = theta[0][0]
    b = theta[1][0]

    return a, b
    
