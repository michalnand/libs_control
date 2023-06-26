import LibsControl
import numpy
import matplotlib.pyplot as plt


if __name__ == "__main__":

    ch         = 5
    amplitude  = numpy.random.rand(ch, 1)
    freq       = 100.0*numpy.random.rand(ch, 1)
    phase      = 2.0*numpy.pi*numpy.random.rand(ch, 1)

    steps = 1000

    x = numpy.arange(steps)/steps


    #signal = amplitude*(numpy.sin(freq*x + phase) > 0.0)
    signal = amplitude*numpy.sin(freq*x + phase)
    signal = signal.sum(axis=0)
    signal = numpy.expand_dims(signal, 1)


    signal_noised = signal + 1.0*numpy.random.randn(signal.shape[0], 1)
    
    signal_denoised = LibsControl.denoising(signal_noised, 10.0, 100)

    snr_noised = (signal**2 + 10**-6)/((signal_noised - signal)**2)
    snr_noised = 10.0*numpy.log10(snr_noised.mean())

    snr_denoised = (signal**2 + 10**-6)/((signal_denoised - signal)**2)
    snr_denoised = 10.0*numpy.log10(snr_denoised.mean())

    print("snr_noised   = ", snr_noised, "dB")
    print("snr_denoised = ", snr_denoised, "dB")

    plt.plot(signal[:, 0],            label = "original", alpha = 0.8, color="red", lw=3.0)
    plt.plot(signal_noised[:, 0],     label = "noised", alpha = 0.4, color="red")
    plt.plot(signal_denoised[:, 0],   label = "denoised", alpha = 0.8, color="deepskyblue", lw=2.0)
    plt.legend()
    plt.show()