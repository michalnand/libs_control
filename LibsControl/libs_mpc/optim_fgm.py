import numpy

"""
    SGD optimizer
    
    Parameters:
    - x_now         : numpy.ndarray, shape (H, M)
    - x_grad        : numpy.ndarray, shape (H, M)
    - learning_rate : float

    Returns:
    - x_result      : numpy.ndarray, shape (H, M)
"""
class OptimFGM:
    def __init__(self, param, learning_rate = 0.001):
        self.learning_rate = learning_rate
        self.y_now         = numpy.zeros_like(param)

    def reset(self):
        self.y_now[:] = 0.0

    def step(self, x_now, x_grad):
        
        x_result, self.y_now = self._optim_step(x_now, x_grad, self.y_now, self.learning_rate)
        return x_result

    def _optim_step(self, x_now, x_grad, y_now, learning_rate, momentum = 0.9):    
        # Update with gradient and apply Nesterov momentum
        y_result = x_now - learning_rate * x_grad
        x_result = y_result + momentum * (y_result - y_now)
        return x_result, y_result