import numpy

class Relay:

    def __init__(self, x_idx, u_idx, x_amp, u_amp):
        self.x_idx = numpy.array(x_idx)
        self.u_idx = numpy.array(u_idx)
        self.x_amp = numpy.array(x_amp)
        self.u_amp = numpy.array(u_amp)

        self.inputs_count = len(self.u_amp)

        self.state = numpy.ones(self.inputs_count, dtype=int)

    def step(self, x):
        for i in range(self.inputs_count):
            if self.state[i] == 1:
                if x[self.x_idx[i], 0] >= self.x_amp[i]:
                    self.state[i] = 0
            else:
                if x[self.x_idx[i], 0] <= -self.x_amp[i]:
                    self.state[i] = 1

        u_result = (self.state == 1)*self.x_amp +  (self.state == 0)*(-1*self.x_amp)

        u_result = numpy.expand_dims(u_result, 1)

        return u_result
