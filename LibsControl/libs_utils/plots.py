import numpy
import matplotlib.pyplot as plt



def plot_poles(re_ol, im_ol, re_cl, im_cl, file_name):
    plt.clf()

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    
    axs.set_xlim([1.2*numpy.min(-numpy.abs([re_ol, re_cl])), 1.2*numpy.max(numpy.abs([re_ol, re_cl]))])
    axs.set_ylim([1.2*numpy.min(-numpy.abs([im_ol, im_cl])), 1.2*numpy.max(numpy.abs([im_ol, im_cl]))])
    axs.scatter(re_ol, im_ol, label="open loop", s = 50)
    axs.scatter(re_cl, im_cl, label="closed loop", s = 25)

    axs.grid(True)
    axs.set_xlabel("real")
    axs.set_ylabel("imag")
    axs.legend()
 
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

def plot_closed_loop_response(t_result, u_result, x_result, file_name, u_labels = None, x_labels = None):
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

        axs[idx].plot(t_result, x_result[:, i], label=lbl, color="purple")
        axs[idx].set_xlabel("time [s]")
        axs[idx].set_ylabel(lbl)
        axs[idx].grid()

        idx+= 1
        
    plt.tight_layout()
    plt.savefig(file_name, dpi = 300)