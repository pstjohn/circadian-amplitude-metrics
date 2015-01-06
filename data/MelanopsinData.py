
import os
this_dir, this_filename = os.path.split(__file__)
import numpy as np
from svg.path import parse_path


file_list = ['mel_p.txt', 'mel_n.txt']




def get_xy(file_name):
    with open(os.path.join(this_dir, file_name)) as f:
        string = f.readline()[:-1]

    path_p = parse_path(string)

    p_list = []
    for part in path_p:
        p_list += [part.start]

    p_list = np.array(p_list)
    x_t = p_list.real
    y_t = p_list.imag

    xmin = x_t.argmin()
    xmax = x_t.argmax()

    # Cut ranges to one pass over the data
    x = x_t[xmin:xmax]
    y = -y_t[xmin:xmax]

    # Scale to the appropriate values




    return x,y

xp, yp = get_xy(file_list[0])
xn, yn = get_xy(file_list[1])

xp += -xp.min()
xp *= 174./xp.max()

yp += -yp[238:350].mean() 
yp *= 1/yp.max()


xn += -xn.min()
xn *= 174./xn.max()
yn += -yn[-1]
yn *= 1/yn.max()

pulse1 = np.array([39., 42.])
pulse2 = np.array([105., 111.])

if __name__ == "__main__":
    import matplotlib.pylab as plt
    from CommonFiles.Utilities import plot_grey_zero

    ax = plt.subplot(111)
    ax.plot(xp, yp, 'r')
    ax.plot(xn, yn, 'b')
    ylim = ax.get_ylim()
    ax.fill_between(pulse1, ylim[0], ylim[1], color='yellow')
    ax.fill_between(pulse2, ylim[0], ylim[1], color='yellow')
    ax.set_ylim(ylim)
    plot_grey_zero(ax)

    plt.show()

