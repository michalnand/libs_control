import numpy
from .sparse_solver  import *
from .non_linear_lib import *


def identification(u, x, dt, steps_count=20, augmentations = []):

    #shift values
    u_now    = u[0:-1, :]    
    x_now    = x[0:-1, :]
    x_next   = x[1:, :]

    dx  = (x_next - x_now)/dt
    
    
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
