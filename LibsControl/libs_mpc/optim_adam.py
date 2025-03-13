import numpy

"""
    Adam optimizer
    
    Parameters:
    - x_now: np.ndarray, shape (H, M)
    - x_grad: np.ndarray, shape (H, M)
    - learning_rate: float
    - m: np.ndarray, shape (H, M), previous first moment vector
    - v: np.ndarray, shape (H, M), previous second moment vector
    - beta1: float, momentum decay rate
    - beta2: float, second moment decay rate
    - epsilon: float, small constant for numerical stability
    - t: int, iteration count for bias correction

    Returns:
    - x_result: np.ndarray, shape (H, M)
    - m_now: np.ndarray, updated first moment vector (H, M)
    - v_now: np.ndarray, updated second moment vector (H, M)
"""
class OptimAdam:

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

    def _optim_step(self, x_now, x_grad, learning_rate, m, v, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
   
        # Update biased first moment estimate
        m_now = beta1 * m + (1 - beta1) * x_grad
        # Update biased second raw moment estimate
        v_now = beta2 * v + (1 - beta2) * (x_grad ** 2)
        
        # Correct bias in first and second moments
        m_hat = m_now / (1 - beta1 ** t)
        v_hat = v_now / (1 - beta2 ** t)
        
        # Update control input adjustment
        x_result = x_now - learning_rate * m_hat / (numpy.sqrt(v_hat) + epsilon)

        return x_result, m_now, v_now