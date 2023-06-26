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


import torch

def denoising(x, alpha = 1.0, steps = 100):

    x_t = torch.from_numpy(x).float()

    x_denoised_t = torch.nn.parameter.Parameter(x_t.clone(), requires_grad=True)
    
    #optimizer = torch.optim.AdamW([x_denoised_t], lr=0.1)

    optimizer = torch.optim.RMSprop([x_denoised_t], lr=0.1)


    for i in range(steps):
        loss_mse = ((x_t - x_denoised_t)**2).mean()

        dif         = x_denoised_t[1:, :] - x_denoised_t[0:-1, :]
        #dif = x_denoised_t - torch.roll(x_denoised_t, -1, dims=[0])
        loss_var    = torch.abs(dif).mean()
        
        loss = loss_mse + alpha*loss_var

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return x_denoised_t.detach().cpu().numpy()

