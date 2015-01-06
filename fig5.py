import numpy as np

from tools.Models.degmodelFinal import model, paramset, y0in
from tools.Amplitude import Amplitude
from tools.Utilities import (color_range, p_integrate, PeriodicSpline,
                             lighten_color)
from tools.PlotOptions import (plot_gray_zero, layout_pad, PlotOptions)

PlotOptions(uselatex=True)

test = Amplitude(model(), paramset, y0in)
test._init_amp_class()

res = 100
spacing = np.linspace(-2, 4, res)
args = [(param, space) for param in xrange(test.NP) for space in
        spacing]
args = np.array(args).reshape((test.NP, res, 2))

out = np.load('fig5_degmodel_arcs.dat')

# pulse1_phi = np.array([ 1.71836561,  2.56379934])
# popt_dict_c = np.vstack([popt_dict['c1'], popt_dict['c2']])
# phases = np.array(popt_dict_c[:,0] + pulse1_phi[1])%(2*np.pi)
# phase_labels = np.array(['c1', 'c2'])
# # phases = (np.array(popt_dict.values())[:,0] + pulse1_phi[1])%(2*np.pi)
# # phase_labels = np.array(popt_dict.keys())
# phase_inds = np.array([np.abs(test.phis - phase).argmin() for phase in
#                        phases])
# state_phase_sort = phase_inds.argsort()
# phase_labels = phase_labels[state_phase_sort]
# phase_inds.sort()


# Minimum of ARCs for each paramter/spacing combo
mins = out[:,:,0,:].min(-1)
# mins = out[:,:,0,phase_inds].min(-1)

real_inds_arr = np.where(~np.isnan(mins))
real_inds = zip(*real_inds_arr)

flat_mins = mins[real_inds_arr[0], real_inds_arr[1], :]

bool_arr_lines = [real_inds_arr[0] == par for par in xrange(test.NP)]
lines = [flat_mins[b] for b in bool_arr_lines]

bool_arr_space = [args[~np.isnan(mins),0] == state for state in
                  xrange(test.NP)]
line_spacing = [args[~np.isnan(mins),1][b] for b in bool_arr_space]


import pylab as plt
# fig = plt.figure()
# ax = fig.add_subplot(111)
# for line, space, label, color in zip(lines, line_spacing, test.plabels,
#                                      color_range(test.NP)):
#     ax.plot(space, line, '.:', label=label, color=color)
# ax.axhline(-0.76127344255888718, ls='--', color='grey')
# ax.legend(loc='best')

# From this plot, we choose a parameter and perturbation strength to
# give the maximum possible amplitude decrease


param = 'vdp'
paramind = test.pdict[param]
scale_index = np.where(spacing ==
         line_spacing[paramind][lines[paramind][:45].argmin()])[0][0]
# scale_index = 51 # Found by argmin() after graphical search

curr_pert = out[paramind, scale_index]

# fig, axmatrix = plt.subplots(nrows=2, ncols=1, sharex=True)
# ax1 = axmatrix[0]
# ax1.plot(test.phis, curr_pert[1], label='PRC')
# ax1.plot(test.phis, curr_pert[2], label='gPRC')
# ax2 = axmatrix[1]
# ax2.plot(test.phis, curr_pert[0], label='ARC')
# # ax2.plot(test.phis[phase_inds], curr_pert[0][phase_inds], 'o')
# # for ind, label in zip(phase_inds, phase_labels):
# #     ax2.annotate(label, (test.phis[ind], curr_pert[0][ind]))
# 
# for ax in axmatrix:
#     ax.legend(loc='best')
#     plot_gray_zero(ax)




# calc responses
import tools.Bioluminescence as bio
from data.MelanopsinData import xn, yn, xp, yp, pulse1, pulse2
from tools.Amplitude import gaussian_phase_distribution

# start both trajectories at the same time
ind_start_xn = yn.argmax()
t_start = xn[ind_start_xn]
ind_start_xp = np.abs(xp - t_start).argmin()

# Scale the data to put time in model periods, such that the model and
# data trajectories will be comparable
data_period = 22.29572274407759
model_period = test.y0[-1]
period_ratio = model_period/data_period

tp = (xp[ind_start_xp:-1]-xp[ind_start_xp])*period_ratio
tn = (xn[ind_start_xn:]-xn[ind_start_xn])*period_ratio

t_end = (tp[-1] + tn[-1])/2

yp = yp[ind_start_xp:-1]
yn = yn[ind_start_xn:]

pulse1 = (pulse1 - t_start)*period_ratio
pulse2 = (pulse2 - t_start)*period_ratio
pulse1_phi = test._t_to_phi(pulse1)
pulse2_phi = test._t_to_phi(pulse2)

positive_class = bio.Bioluminescence(tp, yp)
negative_class = bio.Bioluminescence(tn, yn)

# Smooth and detrend the data using DWT, take y as the sum of the
# circadian range and two closest ranges (seems to match nicely with
# true signal)
for cl in [negative_class, positive_class]:
    cl.x[-1] = t_end
    cl.dwt_breakdown(best_res=2**9)
    cl.y = np.array(cl.dwt['components'][4:7]).sum(0)
    cl.y *= 1/cl.y[0]

# Fit a decaying sinusoid to the control data to estimate period, phase,
# and phase diffusivity
negative_class.fit_sinusoid(weights = np.ones(*negative_class.x.shape))

ts = (negative_class.x * 2 * np.pi)/model_period




phase_of_pulse = test.phis[curr_pert[0].argmin()]
phase_start = (phase_of_pulse - pulse1_phi[1])%(2*np.pi)
delta_p = 2**(spacing[scale_index]) - 1
duration = test._t_to_phi(pulse1[1] - pulse1[0])
std_0 = 0.1


state = 'c1'
state_ind = test.ydict[state]

import pickle
with open('fig5_fit.p', 'rb') as f: popt_dict = pickle.load(f)

origphase, amp, decay = popt_dict[state]
decay *= test.y0[-1]/(2*np.pi)
phase = phase_start


def mel_negative_model():
    test_population = gaussian_phase_distribution(phase, std_0, decay,
                                                  invert_res=60)

    test.phase_distribution = test_population

    y_model = (test.x_bar(ts)[:,state_ind] - test.avg[state_ind])
    return amp*y_model


pulse1_inds = [np.abs(ts - p).argmin() for p in pulse1_phi]
pulse2_inds = [np.abs(ts - p).argmin() for p in pulse2_phi]

def mel_positive_model(p1):
    test_population = gaussian_phase_distribution(phase, std_0, decay,
                                                  invert_res=60)
    test.phase_distribution = test_population

    x_lc = ts
    # Empty solution array
    y_p = np.zeros((ts.shape))

    # Solution up to end of first pulse
    y_p[:pulse1_inds[1]] = test.x_bar(x_lc[:pulse1_inds[1]])[:,state_ind]

    # Create Pulse for first light pulse
    amount = p1*paramset[paramind]
    duration = ts[pulse1_inds[1]] - ts[pulse1_inds[0]]
    pulse_creator = test._p_pulse_creator(paramind, amount, duration)
    try:
        test.calc_pulse_responses(pulse_creator, trans_duration=3)
    except AssertionError:
        test.calc_pulse_responses(pulse_creator, trans_duration=6)

    # Get the phase distribution at the time of the end of the first pulse
    mean_p1 = (phase + x_lc[pulse1_inds[1]])%(2*np.pi)
    std_p1 = np.sqrt(std_0**2 + 2*decay*x_lc[pulse1_inds[1]])

    test_pop_p1 = gaussian_phase_distribution(mean_p1, std_p1, decay)

    # times from 1st to 2nd pulse
    t_pulse = x_lc[pulse1_inds[0]:pulse1_inds[1]] - x_lc[pulse1_inds[1]]
    t_p1_to_p2 = x_lc[pulse1_inds[1]:pulse2_inds[1]] - x_lc[pulse1_inds[1]]


    # Calculate the steady-state trajectory after the first pulse
    pt = test_pop_p1.invert(test.phis, test.prc_single_cell)
    y_p[pulse1_inds[1]:pulse2_inds[1]] = pt.average(t_p1_to_p2,
                                          test.lc_phi(test.phis),
                                          test.phis).T[:,state_ind]

    # Evaluate transients

    # Transient after first pulse
    Delta_x = (test.comb_interp(test.phis, t_p1_to_p2) -
               test.ref_interp(test.phis, t_p1_to_p2))[:,:,state_ind]
    pdf = test_pop_p1(t_p1_to_p2, advance_t=False)
    Averaged_dx = p_integrate(test.phis, (pdf*Delta_x).swapaxes(1,0)).T
    y_p[pulse1_inds[1]:pulse2_inds[1]] += Averaged_dx


    # Transient during first pulse
    x_before = test.comb_interp(test.phis, t_pulse)
    pdf = test_pop_p1.phase_offset(test.phis, 0)[None,:,None]
    avg_xb = (pdf*x_before).swapaxes(1,0)[:,:,state_ind]
    pulse_trans = p_integrate(test.phis, avg_xb)
    y_p[pulse1_inds[0]:pulse1_inds[1]] = pulse_trans


    # Create Pulse for second light pulse
    duration2 = ts[pulse2_inds[1]] - ts[pulse2_inds[0]]
    pulse_creator = test._p_pulse_creator(paramind, amount, duration2)
    try:
        test.calc_pulse_responses(pulse_creator, trans_duration=3)
    except AssertionError:
        test.calc_pulse_responses(pulse_creator, trans_duration=6)


    # Find phase distribution at the start/end of the second pulse
    t_diff = x_lc[pulse2_inds[1]] - x_lc[pulse1_inds[1]]
    mean_p2 = (pt.mu + t_diff)%(2*np.pi)
    std_p2 = np.sqrt(pt.sigma**2 + 2*decay*t_diff)
    test_pop_p2 = gaussian_phase_distribution(mean_p2, std_p2,
                                              decay)

    # times for 2nd pulse
    t_pulse2 = x_lc[pulse2_inds[0]:pulse2_inds[1]] - x_lc[pulse2_inds[1]]
    t_p2_to_end = x_lc[pulse2_inds[1]:] - x_lc[pulse2_inds[1]]

    # Calculate the steady-state trajectory after the second pulse
    pt2 = test_pop_p2.invert(test.phis, test.prc_single_cell)
    y_p[pulse2_inds[1]:] = pt2.average(t_p2_to_end, test.lc_phi(test.phis),
                                  test.phis).T[:,state_ind]

    # Evaluate transients

    # Transient after second pulse
    Delta_x = (test.comb_interp(test.phis, t_p2_to_end) - 
               test.ref_interp(test.phis, t_p2_to_end))[:,:,state_ind]
    pdf = test_pop_p2(t_p2_to_end, advance_t=False)
    Averaged_dx = p_integrate(test.phis, (pdf*Delta_x).swapaxes(1,0)).T
    y_p[pulse2_inds[1]:] += Averaged_dx

    # Transient during  pulse
    x_before = test.comb_interp(test.phis, t_pulse2)
    pdf = test_pop_p2.phase_offset(test.phis, 0)[None,:,None]
    avg_xb = (pdf*x_before).swapaxes(1,0)[:,:,state_ind]
    pulse_trans = p_integrate(test.phis, avg_xb)
    y_p[pulse2_inds[0]:pulse2_inds[1]] = pulse_trans

    # Detrend and scale
    y_p += -test.avg[state_ind]
    y_p *= amp

    return y_p, (test_population, pt, pt2)

yn = mel_negative_model()
yp, populations = mel_positive_model(delta_p)

fig2, axmatrix2 = plt.subplots(nrows=3, sharex=True)
axmatrix2[0].plot(ts, negative_class.y, 'k', label='Control')
axmatrix2[0].plot(ts, positive_class.y, 'r', label='Light Sensitive')

axmatrix2[1].plot(ts, yn, 'k', label='Control (model)')
axmatrix2[1].plot(ts, yp, 'r', label='Light Sensitive (model)')

axpop = axmatrix2[2]
# Population before pulse, at t=6

def plot_pop(population, t_eval, t_plot):
    t_eval = test._t_to_phi(t_eval)
    t_plot = test._t_to_phi(t_plot)
    pop = population(t_eval).squeeze()
    interp = PeriodicSpline(test.phis, pop)
    phi_start = test.phis[pop.argmin()]
    interp_phis = np.linspace(phi_start, phi_start + 2*np.pi, num=100)
    # mean = populations[0].mu + test._t_to_phi(t_eval)
    # interp_phis = np.linspace(mean - np.pi, mean + np.pi, num=100)
    y = interp(interp_phis)
    plot_ts = (interp_phis - phi_start - np.pi + t_plot)
    ts_ind = np.abs(ts - t_plot).argmin()
    return plot_ts, y, ts_ind

# npops = 6
colors = list(color_range(8))

t, y, ts_ind = plot_pop(populations[0], 6., 6.,)
axpop.fill_between(t, 0, y, color=lighten_color(colors[0], 0.5))
axpop.plot(t, y, color=colors[0])
axmatrix2[1].plot(ts[ts_ind], yp[ts_ind], 'o', color=colors[0])

t, y, ts_ind = plot_pop(populations[1], 20. - pulse1[1], 20.)
axpop.fill_between(t, 0, y, color=lighten_color(colors[1], 0.5))
axpop.plot(t, y, color=colors[1])
axmatrix2[1].plot(ts[ts_ind], yp[ts_ind], 'o', color=colors[1])

t, y, ts_ind = plot_pop(populations[1], 41. - pulse1[1], 41.)
axpop.fill_between(t, 0, y, color=lighten_color(colors[2], 0.5))
axpop.plot(t, y, color=colors[2])
axmatrix2[1].plot(ts[ts_ind], yp[ts_ind], 'o', color=colors[2])

t, y, ts_ind = plot_pop(populations[1], 62. - pulse1[1], 62.)
axpop.fill_between(t, 0, y, color=lighten_color(colors[3], 0.5))
axpop.plot(t, y, color=colors[3])
axmatrix2[1].plot(ts[ts_ind], yp[ts_ind], 'o', color=colors[3])

t, y, ts_ind = plot_pop(populations[1], 1, pulse2[1] + 1)
axpop.fill_between(t, 0, y, color=lighten_color(colors[4], 0.5))
axpop.plot(t, y, color=colors[4])
axmatrix2[1].plot(ts[ts_ind], yp[ts_ind], 'o', color=colors[4])

t, y, ts_ind = plot_pop(populations[1], 100. - pulse2[1], 100.)
axpop.fill_between(t, 0, y, color=lighten_color(colors[5], 0.5))
axpop.plot(t, y, color=colors[5])
axmatrix2[1].plot(ts[ts_ind], yp[ts_ind], 'o', color=colors[5])

t, y, ts_ind = plot_pop(populations[1], 120. - pulse2[1], 120.)
axpop.fill_between(t, 0, y, color=lighten_color(colors[6], 0.5))
axpop.plot(t, y, color=colors[6])
axmatrix2[1].plot(ts[ts_ind], yp[ts_ind], 'o', color=colors[6])

t, y, ts_ind = plot_pop(populations[1], 140. - pulse2[1], 140.)
axpop.fill_between(t, 0, y, color=lighten_color(colors[7], 0.5))
axpop.plot(t, y, color=colors[7])
axmatrix2[1].plot(ts[ts_ind], yp[ts_ind], 'o', color=colors[7])

for ax in axmatrix2[:-1]:
    ax.axvspan(pulse1_phi[0], pulse1_phi[1], color='y', alpha=0.5)
    ax.axvspan(pulse2_phi[0], pulse2_phi[1], color='y', alpha=0.5)
    plot_gray_zero(ax)
    ax.legend(loc='best', ncol=2)
    
axmatrix2[2].set_xlim([0, ts[-1]])
axmatrix2[2].set_xticks(2*np.pi*np.arange(7))

labels = [r'$0$']
for i in xrange(1,7): labels += [r'$' + str(2*i) + r'\pi$']

axmatrix2[2].set_xticklabels(labels)
axmatrix2[0].set_title('Normalized Bioluminesence')
axmatrix2[2].set_xlabel(r'$\hat{t}$')
axmatrix2[2].set_ylabel(r'$p(\theta, \hat{t})$')
fig2.tight_layout(**layout_pad)



plt.show()
