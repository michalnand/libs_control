import numpy


"""
    ADOPT optimizer
    
    Parameters:
    - x_now: np.ndarray, shape (H, M)
    - x_grad: np.ndarray, shape (H, M)
    - learning_rate: float
    - m: np.ndarray, shape (H, M), previous first moment vector
    - v: np.ndarray, shape (H, M), previous second moment vector
    - beta1: float, momentum decay rate
    - beta2: float, second moment decay rate
    - epsilon: float, small constant for numerical stability

    Returns:
    - x_result: np.ndarray, shape (H, M)
    - m_now: np.ndarray, updated first moment vector (H, M)
    - v_now: np.ndarray, updated second moment vector (H, M)
"""
class OptimAdopt:
    def __init__(self, param, learning_rate = 0.001):
        self.learning_rate = learning_rate

        self.m  = numpy.zeros_like(param)
        self.v  = numpy.zeros_like(param)

    def reset(self):
        self.m[:] = 0.0
        self.v[:] = 0.0

    def step(self, x_now, x_grad):
        
        x_result, self.m, self.v = self._optim_step(x_now, x_grad, self.learning_rate, self.m, self.v)
        return x_result

    def _optim_step(self, x_now, x_grad, learning_rate, m, v, beta1=0.9, beta2=0.999, epsilon=1e-8):
   
        # update second moment estimate
        v_now = beta2 * v + (1 - beta2) * (x_grad ** 2)
        
        # normalize gradient by the second moment - ADOPT
        normalized_grad = x_grad / numpy.maximum(numpy.sqrt(v_now), epsilon)
        
        # update first moment after normalization
        m_now = beta1 * m + (1 - beta1) * normalized_grad
        
        # Update control input adjustment
        x_result = x_now - learning_rate * m_now

        return x_result, m_now, v_now