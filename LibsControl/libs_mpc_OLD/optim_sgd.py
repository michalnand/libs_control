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
class OptimSGD:
    def __init__(self, param, learning_rate = 0.001):
        self.learning_rate = learning_rate

    def reset(self):
        pass

    def step(self, x_now, x_grad):
        
        x_result = self._optim_step(x_now, x_grad, self.learning_rate)
        return x_result

    def _optim_step(self, x_now, x_grad, learning_rate):    
        x_result = x_now - learning_rate * x_grad
        return x_result