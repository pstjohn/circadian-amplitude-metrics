import numpy as np
import matplotlib
import matplotlib.pylab as plt

from tools.PlotOptions import (PlotOptions, layout_pad, plot_gray_zero,
                               format_2pi_axis)
PlotOptions(uselatex=True)

from tools.Models.simplified_tysonmodel import create_class


from tools.Amplitude import Amplitude
init = create_class()
new = Amplitude(init.model, init.paramset, init.y0)
period = new.y0[-1]

new.limitCycle()
new.roots()
new.findPRC()
new.findARC_whole()

new.__class__ = Amplitude
state_pulse_creator = new._s_pulse_creator(1, 0.5)
new.calc_pulse_responses(state_pulse_creator, trans_duration=3)

ts = new.arc_traj_ts

amounts = [0.5, 1.1, 2.0]

prcs = []
arcs = []
traj = []
refs = []

for amount in amounts:
    s_pulse = new._s_pulse_creator(1, amount)
    new.calc_pulse_responses(s_pulse)
    prcs += [new.prc_single_cell]
    arcs += [new.arc_single_cell]
    traj += [new.traj_interp]
    refs += [new.ref_interp]

tmin = new.phis[arcs[1][:,1].argmin()]
ts = new.arc_traj_ts
    
fig_prc = plt.figure()
gs = matplotlib.gridspec.GridSpec(2,2, width_ratios=[1.8,1])

ax_prc = plt.subplot(gs[0,0])
ax_arc = plt.subplot(gs[1,0], sharex=ax_prc)
ax_state = plt.subplot(gs[0,-1])
ax_traj = plt.subplot(gs[1,-1])

for arc, prc in zip(arcs, prcs):
    ax_prc.plot(new.phis, prc)
    ax_arc.plot(new.phis, arc[:,1])

plot_gray_zero(ax_prc)
plot_gray_zero(ax_arc)

for traji_c, color in zip(traj, ['b', 'g', 'r']):
    traji = traji_c(tmin, ts)
    ax_state.plot(traji[:,0], traji[:,1], ':', color=color, zorder=1)
    ax_state.plot(traji[::10 , 0] , traji[::10 , 1] , '.', color=color,
                  zorder=1)

ax_traj.plot(new.arc_traj_ts, refs[1](tmin, ts)[:,0], 'f')
ax_traj.plot(new.arc_traj_ts, refs[1](tmin, ts)[:,1], 'j')
ax_traj.plot(new.arc_traj_ts, traj[1](tmin, ts)[:,0], 'f--')
ax_traj.plot(new.arc_traj_ts, traj[1](tmin, ts)[:,1], 'j--')

ax_state.plot(new.sol[:,0], new.sol[:,1], 'k', zorder=2, lw=2)
ax_state.plot(new.lc_phi(tmin)[0], new.lc_phi(tmin)[1], 'go', zorder=3)

ax_traj.set_xlim([0, 6*np.pi])
# ax_traj.set_ylim([0, 5])
ax_traj.set_xticks(np.array([0, 1.0, 2.0, 3.0])*2*np.pi)
ax_traj.set_xticklabels([r'$0$', r'$2\pi$',  r'$4\pi$', r'$6\pi$'])
ax_state.set_xticks([0, 0.4, 0.8])
# ax_prc.set_xticks(np.linspace(0, 1.0, 6))

format_2pi_axis(ax_prc)

ax_prc.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax_prc.set_ylim([-np.pi, np.pi])
ax_prc.set_yticklabels([r'$-\pi$', r'$\nicefrac{-\pi}{2}$', r'$0$',
                    r'$\nicefrac{\pi}{2}$', r'$\pi$'])


ax_prc.set_xlabel(r'$\theta$')
ax_prc.set_ylabel(r'$\Delta\theta$')
ax_prc.set_title(r'({\bfseries A}) PRC')

leg = ax_arc.legend([r'$\Delta x = ' + str(a) + '$' for a in amounts],
                    loc=3)
ax_arc.set_xlabel(r'$\theta$')
ax_arc.set_ylabel(r'$\Delta A$')
ax_arc.set_title(r'({\bfseries B}) ARC')

ax_state.set_xlabel(r'$X$')
ax_state.set_ylabel(r'$Y$')
ax_state.set_title(r'({\bfseries C}) State-Space Evolution')

leg = ax_traj.legend([r'$X(\hat{t})$', r'$Y(\hat{t})$'], loc=0)
leg.draw_frame(True)
leg.get_frame().set_alpha(0.5)

ax_traj.set_ylabel('Concentration')
ax_traj.set_xlabel(r'$\hat{t}$')
ax_traj.set_title(r'({\bfseries D}) $\Delta x = 1.1$')


plt.tight_layout(**layout_pad)

plt.show()
