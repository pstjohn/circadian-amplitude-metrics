import numpy as np
import pylab as plt

from tools.Models.simplified_tysonmodel import create_class
from tools.Amplitude import Amplitude
from tools.PlotOptions import (PlotOptions, layout_pad, plot_gray_zero,
                               format_2pi_axis)

PlotOptions(uselatex=True)

new = create_class()
new = Amplitude(new.model, new.paramset, new.y0)


amounts = np.array([1E-1, 1E-2, 1E-3])
prcs = []
arcs = []
for amount in amounts:
    state_pulse = new._s_pulse_creator(1, amount)
    new.calc_pulse_responses(state_pulse)
    prcs += [new.prc_single_cell]
    arcs += [new.arc_single_cell]

prcs = np.array(prcs)
arcs = np.array(arcs)

new.findPRC()
new.findARC_whole()


fig, axmatrix = plt.subplots(nrows=4, sharex=True)
lines = axmatrix[0].plot(new.phis, prcs.T/amounts, '.:')
lines += axmatrix[0].plot(new.phis, new.sPRC[:,1], 'k')
axmatrix[0].legend(lines, [r'$\Delta x(0) = 1E-1$',
                           r'$\Delta x(0) = 1E-2$',
                           r'$\Delta x(0) = 1E-3$',
                           r'$\nicefrac{d}{dx}$'],
                   ncol=2, loc=2)

axmatrix[1].plot(new.phis, arcs[:,:,1].T/amounts, '.:')
axmatrix[1].plot(new.phis, new.sARC[:,1,1], 'k')


prcs = []
arcs = []
amounts = np.array([1., 0.5, 0.1])
d = 0.2
for amount in amounts:
    param_pulse = new._p_pulse_creator(0, amount, d)
    new.calc_pulse_responses(param_pulse)
    prcs += [new.prc_single_cell]
    arcs += [new.arc_single_cell]

prcs = np.array(prcs)
arcs = np.array(arcs)

# fig, axmatrix = plt.subplots(nrows=2, sharex=True)

lines2 = axmatrix[2].plot(new.phis, prcs.T/(amounts*d), '.:')
lines2 += axmatrix[2].plot(new.phis + d/2, new.pPRC[:,0], 'k')

axmatrix[3].plot(new.phis, arcs[:,:,1].T/(amounts*d), '.:')
axmatrix[3].plot(new.phis + d/2, new.pARC[:,1,0], 'k')
axmatrix[2].legend(lines, [r'$\Delta p = 1.0$',
                           r'$\Delta p = 0.5$',
                           r'$\Delta p = 0.1$',
                           r'$\nicefrac{d}{dt\; dp}$'],
                   ncol=2, loc=8)

# axmatrix[0].set_title('Parameter Response')

axmatrix[0].set_ylabel(r'$\nicefrac{\Delta\theta}{\Delta x(0)}$')
axmatrix[1].set_ylabel(r'$\nicefrac{\Delta A}{\Delta x(0)}$')
axmatrix[2].set_ylabel(r'$\nicefrac{\Delta\theta}{d\; \Delta p}$')
axmatrix[3].set_ylabel(r'$\nicefrac{\Delta A}{d\; \Delta p}$')

axmatrix[-1].set_xlabel(r'$\theta$')

for ax in axmatrix:
    plot_gray_zero(ax)
    format_2pi_axis(ax)

fig.tight_layout(**layout_pad)

plt.show()
