import numpy
import matplotlib.pyplot as plt


def get_poles(a, b, k):
    poles_ol = numpy.linalg.eigvals(a) + 0j
    re_ol = poles_ol.real
    im_ol = poles_ol.imag

    poles_cl = numpy.linalg.eigvals(a - b@k) + 0j
    re_cl = poles_cl.real
    im_cl = poles_cl.imag

    return re_ol, im_ol, re_cl, im_cl

def get_poles_mesh(a, b, c, step = 0.05):

    poles_ol = numpy.linalg.eigvals(a) + 0j
    re = poles_ol.real
    im = poles_ol.imag
    
    scale = 1.2*max(numpy.max(numpy.abs(re)), numpy.max(numpy.abs(im)))

 
    x0 = numpy.mgrid[-scale:scale:step, -scale:scale:step]

    ones = numpy.eye(a.shape[0])
                        

    x = numpy.reshape(x0, (2, x0.shape[1]*x0.shape[2])).T
    x = x[:, 1] + 1j*x[:, 0]
    x = numpy.expand_dims(x, axis=1)
    x = numpy.expand_dims(x, axis=2)
    
    x = x*ones
    x = x - numpy.expand_dims(a, axis=0)

    phi = numpy.linalg.pinv(x)

    c_ = numpy.expand_dims(c, axis=0)
    b_ = numpy.expand_dims(b, axis=0)

    y = numpy.matmul(numpy.matmul(c_, phi), b_)

    y = numpy.absolute(y)

    y = numpy.absolute(y).sum(axis=(1, 2))

    y = numpy.reshape(y, (x0.shape[1], x0.shape[2]))

    return scale, y

def plot_poles(re_ol, im_ol, re_cl, im_cl, file_name):
    plt.clf()

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    x_range = max(numpy.abs(numpy.hstack([re_ol, re_cl])))
    y_range = max(numpy.abs(numpy.hstack([im_ol, im_cl])))
    
    axs.set_xlim([-1.2*x_range, 1.2*x_range])
    axs.set_ylim([-1.2*y_range, 1.2*y_range])
    axs.scatter(re_ol, im_ol, label="open loop", s = 50, alpha=1.0)
    axs.scatter(re_cl, im_cl, label="closed loop", s = 25, alpha=1.0)

    for i in range(im_ol.shape[0]):
        axs.annotate(str(i), (re_ol[i], im_ol[i] + 0.3*((i%3) - 1)))

    for i in range(im_cl.shape[0]):
        axs.annotate(str(i), (re_cl[i], im_cl[i] + 0.3*((i%3) - 1)))

    axs.grid(True)
    axs.set_xlabel("real") 
    axs.set_ylabel("imag")
    axs.legend()
 
    plt.tight_layout()
    plt.savefig(file_name, dpi = 300)


def plot_poles_mesh(scale, poles, file_name):
    cr = poles.mean()*20
    poles = numpy.clip(poles, 0, cr)
    
    plt.clf()

    ax = plt.axes(projection ='3d')

    x = 2.0*numpy.arange(0, poles.shape[0])/poles.shape[0] - 1.0
    x = scale*x
    x = numpy.expand_dims(x, axis=1).T
    
    y = 2.0*numpy.arange(0, poles.shape[1])/poles.shape[1] - 1.0
    y = scale*y
    y = numpy.expand_dims(y, axis=1)


    ax.plot_surface(x, y, poles, cmap=plt.get_cmap('hot'))


    ax.set_xlabel("real")
    ax.set_ylabel("imag")

    plt.tight_layout()
    plt.savefig(file_name, dpi = 300)

   


def plot_open_loop_response(t_result, x_result, file_name, labels = None):
    plt.clf()

    count = len(x_result[0])
    fig, axs = plt.subplots(count, 1, figsize=(8, 2*count))

    if count == 1:
        axs = [axs]
    
    idx = 0
    for i in range(len(x_result[0])):
        
        if labels is not None:
            lbl = labels[i]
        else:
            lbl = "x[" + str(i) + "]"

        axs[idx].plot(t_result, x_result[:, i], label=lbl, color="deepskyblue")
        axs[idx].set_xlabel("time [s]")
        axs[idx].set_ylabel(lbl)
        axs[idx].grid()

        idx+= 1
        
    plt.tight_layout()
    plt.savefig(file_name, dpi = 300)

def plot_closed_loop_response(t_result, u_result, x_result, x_hat = None, file_name = "output.png", u_labels = None, x_labels = None):
    plt.clf()

    count = len(u_result[0]) + len(x_result[0])
    fig, axs = plt.subplots(count, 1, figsize=(8, 2*count))
    

    idx = 0
    for i in range(len(u_result[0])):
        
        if u_labels is not None:
            lbl = u_labels[i]
        else:
            lbl = "u[" + str(i) + "]"

        axs[idx].plot(t_result, u_result[:, i], label=lbl, color="deepskyblue")
        axs[idx].set_xlabel("time [s]")
        axs[idx].set_ylabel(lbl)
        axs[idx].grid()

        idx+= 1

    for i in range(len(x_result[0])):
        
        if x_labels is not None:
            lbl = x_labels[i]
        else:
            lbl = "x[" + str(i) + "]"

        if x_hat is not None:

            axs[idx].plot(t_result, x_hat[:, i], label= "estimated", color="purple", alpha=0.75)
            axs[idx].plot(t_result, x_result[:, i], label= "measured", color="deepskyblue", alpha=0.75)

        else:
            axs[idx].plot(t_result, x_result[:, i], label=lbl, color="deepskyblue", alpha=0.75)
        
        axs[idx].set_xlabel("time [s]")
        axs[idx].set_ylabel(lbl)

        if x_hat is not None: 
            axs[idx].legend()
        axs[idx].grid()

        idx+= 1
        
    plt.tight_layout()
    plt.savefig(file_name, dpi = 300)
