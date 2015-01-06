import numpy as np
import matplotlib.pylab as plt

from tools.PlotOptions import PlotOptions, layout_pad, plot_gray_zero
PlotOptions(uselatex=True)

from tools.Models.simplified_tysonmodel import create_class
from tools.Utilities import lighten_color


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
phi_start1 = new._t_to_phi(new.Tmax[0] + 0.2)
phi_start2 = new._t_to_phi(new.Tmax[0] - 0.2)

traj1 = new.traj_interp(phi_start1, ts)
traj2 = new.traj_interp(phi_start2, ts)
ref1 = new.ref_interp(phi_start1, ts)
ref2 = new.ref_interp(phi_start2, ts)

traj1_short = traj1[:50]
traj2_short = traj2[:50]
ref1_short  = ref1[:50]
ref2_short  = ref2[:50]


x = np.linspace(0, 0.8, 100)
y = np.linspace(0, 5.0, 100)
xx, yy = np.meshgrid(x, y)
u = np.zeros(xx.shape)
v = np.zeros(xx.shape)

for i in xrange(100):
    xy_row = np.array([xx[i], yy[i]]).T

    ydot_row = new.dydt(xy_row)
    u[i] = ydot_row[:,0]
    v[i] = ydot_row[:,1]


speed = np.sqrt((u/u.max())**2 + (v/v.max())**2)    
amp1 = ((traj1[:150] - new.avg)**2
        - (ref1[:150] - new.avg)**2)[:,1]
amp2 = ((traj2[:150] - new.avg)**2
        - (ref2[:150] - new.avg)**2)[:,1]


fig = plt.figure(figsize=[3.425, 1.4])
xlim = [0., 0.8]
ylim = [0., 5.]
ax_traj = fig.add_subplot(131, aspect=(xlim[1]/ylim[1]))
ax_traj.set_xlim(xlim)
ax_traj.set_ylim(ylim)

ax_traj.plot(new.sol[:,0], new.sol[:,1], 'k', lw=2)

ax_traj.plot(traj1_short[: , 0] , traj1_short[: , 1], ':', lw=1 ,
             color='r' , zorder=1)
ax_traj.plot(traj1_short[::10 , 0] , traj1_short[::10 , 1] , '.'  ,
             color='r' , zorder=1)
ax_traj.plot(traj2_short[: , 0] , traj2_short[: , 1] , ':', lw=1 ,
             color='b' , zorder=1)
ax_traj.plot(traj2_short[::10 , 0] , traj2_short[::10 , 1] , '.'  ,
             color='b' , zorder=1)
ax_traj.plot(new.sol[:,0], new.sol[:,1], 'k', lw=2)

y0_1 = new.lc_phi(phi_start1)
y0_2 = new.lc_phi(phi_start2)
ax_traj.plot(y0_1[0], y0_1[1], 'go')
ax_traj.plot(y0_2[0], y0_2[1], 'go')

ax_comp_1 = fig.add_subplot(132, sharey=ax_traj)
ax_comp_amp = fig.add_subplot(133, sharex=ax_comp_1)

# plot_gray_zero(ax_comp_amp, zorder=2, linewidth=1.25)

ax_comp_amp.fill_between(new.arc_traj_ts[:150],
                         np.zeros(amp1.shape), amp1,
                         color=lighten_color('r',0.5),
                         facecolor=lighten_color('r', 0.8), zorder=1)
ax_comp_amp.fill_between(new.arc_traj_ts[:150],
                         np.zeros(amp2.shape), amp2,
                         color=lighten_color('b', 0.5),
                         facecolor=lighten_color('b', 0.8), zorder=1)


ax_comp_amp.set_ylim([-2.5, 2.5])
# ax_comp_amp.set_ylim([-new.avg[1], -new.avg[1]+ylim[1]])

ax_comp_1.plot(ts[:150], traj1[:150, 1], 'r--', zorder=2)
ax_comp_1.plot(ts[:150], traj2[:150, 1], 'b--', zorder=2)
ax_comp_1.plot(ts[:150], ref1[:150 , 1], 'k'  , zorder=2)
ax_comp_1.axhline(y=new.avg[1], linestyle='--', color='grey')

ax_comp_1.set_yticks(range(5))
ax_comp_1.set_xlim([0, 3*np.pi])
ax_comp_1.set_xticks(np.linspace(0, 3*np.pi, 4))
ax_comp_1.set_xticklabels([r'$0$', r'$\pi$',  r'$2\pi$', r'$3\pi$'])
# ax_comp_1.set_xlim([0,1.5])

ax_traj.set_title(r'({\bfseries B}) State-space')
ax_traj.set_ylabel('$Y$')
ax_traj.set_xlabel('$X$')

ax_comp_1.set_title(r'({\bfseries C}) Time-series')
ax_comp_1.set_xlabel(r'$\hat{t}$')
ax_comp_1.text(8., 1.6, r'$\mu$')

ax_comp_amp.set_title(r'({\bfseries D}) $h(\hat{t})$')
ax_comp_amp.set_xlabel(r'$\hat{t}$')


fig.tight_layout(**layout_pad)

plt.show()
