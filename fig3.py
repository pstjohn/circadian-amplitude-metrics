import numpy as np
import matplotlib.pylab as plt

from tools.Amplitude import (Amplitude, gaussian_phase_distribution,
                             mean_std, wrapped_gaussian)
from tools.Models.simplified_tysonmodel import create_class
from tools.Utilities import (PeriodicSpline, color_range)
from tools.PlotOptions import PlotOptions, layout_pad

PlotOptions(uselatex=True)

init = create_class()
test = Amplitude(init.model, init.paramset, init.y0)

state_pulse_creator = test._s_pulse_creator(1, 0.65)
test.calc_pulse_responses(state_pulse_creator, trans_duration=3)

mean = 0
std = 0.5
decay = test.y0[-1]*0.1/(2*np.pi)
phis = np.linspace(0, 2*np.pi, num=100, endpoint=True)
po = wrapped_gaussian(phis, mean, std)
po_interp = PeriodicSpline(phis, po)

test_population = gaussian_phase_distribution(mean, std, decay,
                                              invert_res=60)
perturbed_popul = test_population.invert(test.phis,
                                         test.prc_single_cell,
                                         phi_offset=0)

test.calc_population_responses(test_population, tarc=False)

mean_p, std_p = mean_std(test.z_hat(0))
mean_pert_pop = gaussian_phase_distribution(mean_p, std_p, decay)

def expdist(x, d):
    return (np.exp(d*x) - 1)/(np.exp(d) - 1)

trange = (expdist(np.linspace(0, 1, num=8),
                  1.25)*(7.5*np.pi))
crange = list(color_range(len(trange)))

xlim = np.array([-np.pi/2, 8*np.pi+np.pi/2])

fig, axmatrix = plt.subplots(nrows=2, sharex=True)

ax1 = axmatrix[0]
for t, color in zip(trange, crange):
    pop = perturbed_popul(t, phi_res=len(phis)).squeeze()
    interp = PeriodicSpline(phis, pop)

    pop_mean = mean_pert_pop(t, phi_res=len(phis)).squeeze()
    interp_mean = PeriodicSpline(phis, pop_mean)

    mean = perturbed_popul.mu + t
    plot_phis = np.linspace(mean - np.pi, mean + np.pi, num=100)
    y = interp(plot_phis) 
    ax1.fill_between(plot_phis, 0, y, color=color, alpha=0.5)

    if t == trange[0]:
        ax1.plot(plot_phis, y, color=color, label=r'$p(\theta, \hat{t})$')
        ax1.plot(plot_phis, interp_mean(plot_phis), '--', color=color,
                 label=r'$\tilde{p}(\theta, \hat{t})$')
    else:
        ax1.plot(plot_phis, y, color=color)
        ax1.plot(plot_phis, interp_mean(plot_phis), '--', color=color)

ax1.legend(loc='best')
ax1.set_xlim(xlim)
ax1.set_ylim([0, 2])


ax2 = axmatrix[1]
# Attach the phase distribution
test.phase_distribution = perturbed_popul

ts = np.linspace(xlim[0], xlim[1], num=600)

x_bar = test.x_bar(ts)[:,1]
x_lc = test.lc_phi(ts)[:,1]

ax2.plot(ts, x_lc, 'k--', label=r'$x^\gamma(\theta)$')
ax2.plot(ts, x_bar, 'b', label=r'$\bar{x}(\hat{t})$')
for t, color in zip(trange, crange):
    ax2.plot(t, test.x_bar(t)[:,1], 'o', color=color)

ax2.set_xticks(np.arange(0,10,2)*np.pi)
ax2.set_xticklabels(['0'] +
                   [r'$' + str(x) + r'\pi$' for x in range(2,10,2)])
ax2.set_xlabel(r'$\hat{t}$')

# ax1.set_ylabel(r'$p(\theta, t)$')
leg = ax2.legend(loc=1)
leg.draw_frame(True)
leg.get_frame().set_alpha(0.5)




fig.tight_layout(**layout_pad)


plt.show()
