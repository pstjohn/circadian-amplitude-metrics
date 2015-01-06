import numpy as np
import matplotlib.pylab as plt

from tools.Amplitude import (Amplitude, gaussian_phase_distribution)

from tools.Models.degmodelFinal import create_class

from tools.PlotOptions import (PlotOptions, layout_pad, plot_gray_zero,
                               format_2pi_axis)

PlotOptions(uselatex=True)


init = create_class()
test = Amplitude(init.model, init.paramset, init.y0)


test.findPRC()
test.findARC_whole()

stateind = 5
paramind = test.pdict['vtp']
amount   = 0.2 * init.paramset[paramind]
duration = np.pi/2

param_pulse = test._p_pulse_creator(paramind, amount, duration)
test.calc_pulse_responses(param_pulse, trans_duration=4)

mu    = 0
sigma = 0.5
decay = 0.02*test.y0[-1]/(2*np.pi)
# decay = 0.05
# decay = 5E-3 * test.y0[-1] / (2*np.pi)

initial_population = gaussian_phase_distribution(mu, sigma, decay)

test.calc_population_responses(initial_population, tarc=False)

# arc_interp = PeriodicSpline(test.phis, test.pARC[:,stateind, paramind])
# arc_moved = arc_interp(test.phis + duration/2)

phi_min_arc = test.phis[test.arc_single_cell[:,stateind].argmin()]
phi_min_pop = test.phis[test.arc_population.argmin()]
phi_max = test.phis[test.arc_population.argmax()]



# mu = (phi_min_pop + phi_min_arc)/2.

# ts = np.linspace(test.comb_interp.ts.min(), 5*test.y0[-1], num=300)
ts = test.comb_interp.ts

ax_prc = plt.subplot2grid((3,2), (0,0))
ax_arc = plt.subplot2grid((3,2), (0,1), sharex=ax_prc)
ax_ts  = plt.subplot2grid((3,2), (1,0), colspan=2)
ax_pop = plt.subplot2grid((3,2), (2,0), colspan=2, sharex=ax_ts, sharey=ax_ts)

fig = plt.gcf()

format_2pi_axis(ax_prc, y=True)
ax_prc.set_xlabel(r'$\theta$')
ax_prc.set_ylabel(r'$\hat{\theta}$')
plot_gray_zero(ax_prc)
format_2pi_axis(ax_arc, y=False)
ax_arc.set_xlabel(r'$\theta$')
plot_gray_zero(ax_arc)

ax_prc.set_title('PRC')
ax_arc.set_title('ARC')

ax_prc.axvline(phi_min_arc, linestyle='--', color='r')
ax_arc.axvline(phi_min_arc, linestyle='--', color='r')
ax_prc.axvline(phi_min_pop, linestyle='--', color='c')
ax_arc.axvline(phi_min_pop, linestyle='--', color='c')

ax_prc.plot(test.phis, test.prc_single_cell, label='Single Cell')
ax_prc.plot(test.phis, test.prc_population, label='Population')

ax_prc.legend(loc='best', ncol=2)

rescale = lambda x: x/x.std()
ax_arc.plot(test.phis, rescale(test.arc_single_cell[:,stateind]),
            label='Single Cell')
# ax_arc.plot(test.phis, test.tarc_population[:,stateind],
#             label='Population')
ax_arc.plot(test.phis, rescale(test.arc_population), 'g', label='Desync')
ax_arc.legend(loc='best', ncol=2)


initial_population = gaussian_phase_distribution(phi_min_arc, sigma, decay)
test.calc_population_responses(initial_population, tarc=False)
perturbed_popul = initial_population.invert(test.phis,
                                            test.prc_single_cell,
                                            phi_offset=0)
mean_p, std_p = perturbed_popul.mu, perturbed_popul.sigma
mean_pert_pop = gaussian_phase_distribution(mean_p, std_p, decay)

ax_ts.plot(ts, test.x_bar(ts)[:,stateind] - test.avg[stateind], 'b:', label=r'$\bar{x}(t)$')
ax_ts.plot(ts, test.x_hat(ts)[:,stateind] - test.avg[stateind], 'r', label=r'$\hat{x}(t)$')
ax_ts.plot(ts, test.x_hat_ss(ts)[:,stateind] - test.avg[stateind], 'g:', label=r'$\hat{x}_{ss}(t)$')

ax_ts.legend(loc='best', ncol=3)

plt.setp(ax_ts.get_xticklabels(), visible=False)

initial_population = gaussian_phase_distribution(phi_min_pop, sigma,
                                                 decay)
test.calc_population_responses(initial_population, tarc=False)
perturbed_popul = initial_population.invert(test.phis,
                                            test.prc_single_cell,
                                            phi_offset=0)
mean_p, std_p = perturbed_popul.mu, perturbed_popul.sigma
mean_pert_pop = gaussian_phase_distribution(mean_p, std_p, decay)
ax_pop.plot(ts, test.x_bar(ts)[:,stateind] - test.avg[stateind], 'b:',
            label=r'$\bar{x}(\hat{t})$')
ax_pop.plot(ts, test.x_hat(ts)[:,stateind] - test.avg[stateind], 'r',
            label=r'$\hat{x}(\hat{t})$')
ax_pop.plot(ts, test.x_hat_ss(ts)[:,stateind] - test.avg[stateind],
            'g:', label=r'$\hat{x}_{ss}(\hat{t})$')


ax_ts.set_xlim([-np.pi/2, 8*np.pi])
ax_ts.set_xticks(np.pi*np.arange(0,10,2))
ax_pop.set_xlim([-np.pi/2, 8*np.pi])
ax_pop.set_xticks(np.pi*np.arange(0,10,2))
ax_pop.set_xticklabels([r'$0$', r'$2\pi$', r'$4\pi$', r'$6\pi$', r'$8\pi$'])




# def expdist(x, d):
#     return (np.exp(d*x) - 1)/(np.exp(d) - 1)
# 
# trange = (expdist(np.linspace(0, 1, num=7),
#                   1.25)*(7.0*np.pi))*test.y0[-1]/(2*np.pi)
# # trange = trange[:-1]
# crange = list(color_range(len(trange)))
# 
# for t, color in zip(trange, crange):
#     pop = perturbed_popul(t, phi_res=len(test.phis)).squeeze()
#     interp = PeriodicSpline(test.phis, pop)
# 
#     pop_mean = mean_pert_pop(t, phi_res=len(test.phis)).squeeze()
#     interp_mean = PeriodicSpline(test.phis, pop_mean)
#     
#     phi_mid = test.phis[pop.argmax()]
#     interp_phis = np.linspace(phi_mid - np.pi, phi_mid + np.pi, num=100)
#     y = interp(interp_phis)
#     plot_ts = test._phi_to_t(interp_phis - phi_mid) + t
#     ax_pop.fill_between(plot_ts, 0, y, color=color, alpha=0.5)
# 
#     lab1 = 'r$p(\theta, t)$'
#     ax_pop.plot(plot_ts, y, color=color)
#     ax_pop.plot(plot_ts, interp_mean(interp_phis), '--', color=color)
# 
# 
# ax_pop.set_ylim([0, 3])
# ax_pop.set_ylabel(r'$p(\theta, t)$')

ax_pop.set_xlabel(r'$\hat{t}$')

fig.tight_layout(**layout_pad)

# plot(ts, test.x_bar(ts)[:,stateind])
# plot(ts, test.x_hat(ts)[:,stateind])
# plot(ts, test.x_hat_ss(ts)[:,stateind])
#
plt.show()
