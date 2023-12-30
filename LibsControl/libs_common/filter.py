import numpy
import matplotlib.pyplot as plt


class BiquadFilter:

    def __init__(self, b0, b1, b2, a1, a2, quantization = None):

        self.b0 = b0
        self.b1 = b1
        self.b2 = b2

        self.a1 = a1
        self.a2 = a2

        self.x2 = 0
        self.x1 = 0

        self.y2 = 0
        self.y1 = 0

        if quantization is not None:

            if quantization == "int8":
                max_range = 127.0
            else:
                max_range = 32767.0

            coeffs  = numpy.array([b0, b1, b2, a1, a2])
            k = numpy.abs(numpy.max(coeffs))
            scaling = max_range/k

            self.b0_quant = int(max_range*b0/k)
            self.b1_quant = int(max_range*b1/k)
            self.b2_quant = int(max_range*b2/k)

            self.a1_quant = int(max_range*a1/k)
            self.a2_quant = int(max_range*a2/k)

            self.b0 = round(self.b0_quant/scaling, 6)
            self.b1 = round(self.b1_quant/scaling, 6)
            self.b2 = round(self.b2_quant/scaling, 6)

            self.a1 = round(self.a1_quant/scaling, 6)
            self.a2 = round(self.a2_quant/scaling, 6)

            
            print(f"{self.b0:05}", "\t", self.b0_quant)
            print(f"{self.b1:05}", "\t", self.b1_quant)
            print(f"{self.b2:05}", "\t", self.b2_quant)

            print(f"{self.a1:05}", "\t", self.a1_quant)
            print(f"{self.a2:05}", "\t", self.a2_quant)
            

        else:
            self.scaling = 1.0
            
            


    def step(self, x):

        y = self.b0*x + self.b1*self.x1 + self.b2*self.x2 + self.a1*self.y1 + self.a2*self.y2

        self.x2 = self.x1
        self.x1 = x

        self.y2 = self.y1
        self.y1 = y

        return y



    def step_response(self, points = 500):
        x_result = []
        y_result = []
        for i in range(points):

            if i > points*0.1:
                x = 1.0
            else:
                x = 0.0

            y = self.step(x)

            x_result.append(x)
            y_result.append(y)

        return x_result, y_result

    def freq_response(self, points = 44100, f_sampling = 1):
        
        result_f = []
        result_a = []
        for i in range(points):
            z     = numpy.exp(2.0j*numpy.pi*i/points + (10**-12))

            nom, denom = self._transfer_function(z)

            h = nom/denom

            result_f.append(f_sampling*i/points)
            result_a.append(abs(h))

        return numpy.array(result_f), numpy.array(result_a)

    def poles_zeros(self, step = 0.01):

        x = numpy.mgrid[-1.0:1.0:step, -1.0:1.0:step]

        z = x[1] + 1.0j*x[0]

        zeros_idx    = numpy.where(z == 0)

        z[zeros_idx] = 10**-12
        nom, denom = self._transfer_function(z)

        amp = numpy.log10(numpy.abs(nom/denom) + 10**-3)

        return amp

    def _transfer_function(self, z):

            
        nom   = (self.b0 + self.b1*(z**-1) + self.b2*(z**-2))
        denom = (1 - self.a1*(z**-1) - self.a2*(z**-2))

        return nom, denom


if __name__ == "__main__":

    fs = 1500

    a1 = 1.52820817
    a2 = -0.88896918

    b0 = 0.05551541
    b1 = 0.00000000e+0
    b2 = -0.05551541
 
    
    fil = BiquadFilter(b0, b1, b2, a1, a2)
    fil_quant16 = BiquadFilter(b0, b1, b2, a1, a2, "int16")
    fil_quant8 = BiquadFilter(b0, b1, b2, a1, a2, "int8")

    x_result, y_result          = fil.step_response()
    freq_result, freq_response  = fil.freq_response(f_sampling=fs)
    _, freq_response_quant16    = fil_quant16.freq_response(f_sampling=fs)
    _, freq_response_quant8     = fil_quant8.freq_response(f_sampling=fs)

    poles_zeros = fil.poles_zeros()
    
    fig, ax = plt.subplots(3, figsize=(6,8))

    ax[0].plot(x_result, label="input")
    ax[0].plot(y_result, label="response")
    ax[0].grid()
    ax[0].set_title("step response")
    
    ax[1].plot(freq_result, freq_response, label="float", linewidth=3)
    ax[1].plot(freq_result, freq_response_quant16, label="int16", linewidth=1)
    ax[1].plot(freq_result, freq_response_quant8,  label="int8", linewidth=1)
    ax[1].grid()
    ax[1].legend(loc='upper center')
    ax[1].set_xlabel("frequency [Hz]")
    ax[1].set_ylabel("magnitude")
    ax[1].set_title("frequency response")

    circle = plt.Circle((0, 0), 1.0, color='b', fill=False)
    img = ax[2].imshow(poles_zeros, cmap='hot', interpolation='nearest',extent=[-1, 1, -1, 1])
    plt.colorbar(img, ax=ax[2])
    ax[2].add_patch(circle)
    ax[2].set_xlabel("Real")
    ax[2].set_ylabel("Imag")
    ax[2].set_title("poles and zeros [log]")
    
    

    plt.tight_layout()

    k = 0.2

    plt.text(-4, 0.5 - k*0, "a1 = " + str(round(a1, 5)) + "  " + str(fil_quant16.a1_quant) + "  " + str(fil_quant8.a1_quant), fontweight="bold")
    plt.text(-4, 0.5 - k*1, "a2 = " + str(round(a2, 5)) + "  " + str(fil_quant16.a2_quant) + "  " + str(fil_quant8.a2_quant), fontweight="bold")

    plt.text(-4, 0.5 - k*3, "b0 = " + str(round(b0, 5)) + "  " + str(fil_quant16.b0_quant) + "  " + str(fil_quant8.b0_quant), fontweight="bold")
    plt.text(-4, 0.5 - k*4, "b1 = " + str(round(b1, 5)) + "  " + str(fil_quant16.b1_quant) + "  " + str(fil_quant8.b1_quant), fontweight="bold")
    plt.text(-4, 0.5 - k*5, "b2 = " + str(round(b2, 5)) + "  " + str(fil_quant16.b2_quant) + "  " + str(fil_quant8.b2_quant), fontweight="bold")

    plt.text(-4, 0.5 - k*7, "f_sampling = " + str(fs) + "[Hz]", fontweight="bold")
    plt.show()

