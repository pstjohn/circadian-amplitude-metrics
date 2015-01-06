import numpy as np
import cPickle as pickle
import pylab as plt

from tools.Utilities import lighten_color

# LOAD AND PROCESS STOCHASTIC TRAJECTORIES
# ========================================

data = pickle.load(open('figS2_stochastic_data.p', 'rb'))

# pulse   = data['pulse']
ts_cont = data['ts_cont'] # t = 0 defined as end of pulse
ys_cont = data['ys_cont']
ts_pert = data['ts_pert']
ys_pert = data['ys_pert']


# ESTIMATE SINUSOIDAL PARAMETERS FROM STOCHASTIC DATA
# ===================================================

state = 1 # State used for estimation and plotting (Protein)

from tools.DecayingSinusoid import DecayingSinusoid
master = DecayingSinusoid(ts_cont[ts_cont > 0],
                          ys_cont.mean(0)[ts_cont > 0, state],
                          max_degree=0).run()
decay     = master.averaged_params['decay'].value
amplitude = master.averaged_params['amplitude'].value
phase     = master.averaged_params['phase'].value
period    = master.averaged_params['period'].value

phis_cont = 2*np.pi*ts_cont/period
phis_pert = 2*np.pi*ts_pert/period


# SET UP CONTINUOUS APPROXIMATION
# ===============================

# Import deterministic model
from tools.Models.simplified_tysonmodel_2 import create_class
model = create_class()
model.average()

# Set up pdf-based model
from tools.Amplitude import Amplitude, gaussian_phase_distribution
ampl = Amplitude(model.model, model.paramset, model.y0)
par_ind = ampl.pdict['kt']
param_pulse_creator = ampl._p_pulse_creator(1, -10., np.pi/4)
ampl.calc_pulse_responses(param_pulse_creator, trans_duration=4)


# ESTIMATE INITIAL POPULATION
# ===========================

# Find matching distribution for control at t = 0 (pulse start)
# First find expected amplitude/phase if p(\theta) has sigma=0,
# then determine amount amount the sinusoid has been damped/shifted
init_amps, init_phases, baselines = ampl._cos_components()
sigma = np.sqrt(-2*np.log(amplitude/init_amps[state]))
mu = phase - init_phases[state]%(2*np.pi)


# CALCULATE DETERMINISTIC CONTROL AND PERTURBED TRAJECTORIES
# ==========================================================

# set up initial population
test_population = gaussian_phase_distribution(mu, sigma, decay,
                                              invert_res=60)

# Calculate desynchronization from perturbation with given
# population
ampl.calc_population_responses(test_population, tarc=False)

# Find control and perturbed trajectories
phis_det = ampl.comb_interp.ts

x_bar = ampl.x_bar(phis_det)[:, state]
x_hat = ampl.x_hat(phis_det)[:, state]


# PLOT STOCHASTIC AND DETERMINISTIC TRAJECTORIES
# ==============================================

from tools.PlotOptions import PlotOptions, layout_pad
PlotOptions(uselatex=True)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.axvspan(-np.pi/4, 0, color=lighten_color('y', 0.5))
ax.plot(phis_cont, ys_cont.mean(0)[:,state], 'k',
        label='Unperturbed Population', zorder=2)
ax.plot(phis_det, x_bar, 'k-.', label=r'$\bar{x}(\hat{t})$', zorder=2)

ax.plot(phis_pert, ys_pert.mean(0)[:,1], 'r', zorder=1,
        label='Perturbed Population')
ax.plot(phis_det, x_hat, 'r-.', label=r'$\hat{x}(\hat{t})$', zorder=1)

ax.set_xlim([-np.pi, 6*np.pi])
ax.set_xticks([-np.pi, 0, 2*np.pi, 4*np.pi, 6*np.pi])
ax.set_xticklabels([r'$-\pi$', r'$0$', r'$2\pi$', r'$4\pi$', r'$6\pi$'])

ax.legend(loc='best', ncol=2)

ax.set_ylabel(r'$Y$')

from tools.Utilities import color_range, PeriodicSpline
## Population plots
def expdist(x, d):
    return (np.exp(d*x) - 1)/(np.exp(d) - 1)

trange = 4*1.15*np.pi*expdist(np.linspace(0,1,4), 0.5) + np.pi/2
# trange = np.arange(4)*1.5*np.pi + 2*np.pi/4
crange = list(color_range(len(trange)+1))

phis = ampl.phis
perturbed_popul = ampl.phase_distribution.invert(phis, ampl.prc_single_cell)

ax_pop = fig.add_subplot(212, sharex=ax)

def plot_pop(population, phi_plot):
    pop = population(phi_plot).squeeze()
    interp = PeriodicSpline(ampl.phis, pop)
    phi_start = ampl.phis[pop.argmin()]
    interp_phis = np.linspace(phi_start, phi_start + 2*np.pi, num=100)
    y = interp(interp_phis)
    plot_ts = (interp_phis - phi_start - np.pi + phi_plot)
    phis_ind = np.abs(phis_pert - phi_plot).argmin()

    # check stochastic oscillator
    points = ys_pert[phis_ind,:,:]
    phases = []
    for point in points:
        phases += [model.phase_of_point(point)[0]]

    hist, bin_edges = np.histogram(phases, bins=15)
    bin_width = bin_edges[1] - bin_edges[0] 
    inds = bin_edges[:-1] < (phi_start)
    bins = bin_edges[:-1][~inds].tolist() + (bin_edges[:-1][inds] +
                                             bin_edges[-1]).tolist()

    hist = hist[~inds].tolist() + hist[inds].tolist()
    bins = np.array(bins) + phi_plot - np.pi - phi_start

    return plot_ts, y, phis_ind, hist/(225. * bin_width), bins, bin_width

t, y, phis_ind, hist, bins, bin_width = plot_pop(ampl.phase_distribution,
                                           -np.pi/2)
ax_pop.fill_between(t, 0, y, color=lighten_color(crange[0], 0.5))
ax_pop.plot(t, y, color=crange[0], lw=1.25)
ax.plot(phis_pert[phis_ind], ys_pert[phis_ind,:,1].mean(0), 'o',
        color=crange[0])
ax_pop.bar(bins, hist, bin_width, facecolor=lighten_color(crange[0], 0.5))

for phi_plot, color in zip(trange, crange[1:]):
    t, y, phis_ind, hist, bins, bin_width = plot_pop(perturbed_popul,
                                                     phi_plot)
    ax_pop.fill_between(t, 0, y, color=color, alpha=0.5)
    ax_pop.plot(t, y, color=color, lw=1.25)
    ax.plot(phis_pert[phis_ind], ys_pert[phis_ind,:,1].mean(0), 'o',
            color=color)
    ax_pop.bar(bins, hist, bin_width, facecolor=lighten_color(color, 0.5))



ax_pop.set_xlim([-np.pi, 6*np.pi])
ax_pop.set_xlabel(r'$\hat{t}$')
ax_pop.set_ylabel(r'$p(\theta, \hat{t})$')
fig.tight_layout(**layout_pad)


plt.show()
