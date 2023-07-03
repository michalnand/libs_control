import torch
import numpy

def denoising(x, alpha = 1.0, steps = 100):

    x_t = torch.from_numpy(x).float()


    x_denoised_t = torch.nn.parameter.Parameter(x_t.clone(), requires_grad=True)
    

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


def smoothing(x, alpha = 0.1):

    y_result = x.copy()

    for i in range(1, x.shape[0]):

        y_result[i, :] = (1.0 - alpha)*y_result[i-1, :] + alpha*x[i, :]

    return y_result
