import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker
from .utils import linear_fit

def plot_acf_2d(time, r, acf_t, ax):
    #fig, ax = plt.subplots(figsize=[3, 2.25])
    x, y = np.meshgrid(time,r)
    Z = np.array(acf_t).T
    vmax = 3*acf_t[0][-1]
    norm = cm.colors.Normalize(vmax=vmax, vmin=0)
    levels = np.linspace(0.0, vmax, 11)
    pos = ax.contourf(
        x, y, Z,
        levels = levels,
        norm=norm,
        #extent=(x.min(), x.max(), y.min(), y.max()), 
        cmap='Blues', 
        extend = 'max',
        #vmin=0, 
        #vmax=0.001, 
        #interpolation='nearest'
    )

    ax.set_ylim(top=3.5)
    ax.set_ylabel(r'r ($\AA{}$)')
    ax.set_xlabel('time (ps)')
    cbar = plt.colorbar(pos, ax=ax,pad=-0.2)
    cbar.ax.set_yticklabels([])
    cbar.ax.set_ylabel(r'G$_{d}(r,t)$', rotation=270)

def plot_EV(time, E_atom, V, ax1):
    #ax1.set_aspect(1)

    color = 'tab:red'
    ax1.set_xlabel('time (ps)')
    ax1.set_ylabel('total energy (eV/atom)', color=color)
    ax1.plot(time, E_atom, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(r'volume (nm$^3$)', color=color)  # we already handled the x-label with ax1
    ax2.plot(time, V/1000, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(25))

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.03))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

def plot_MSD(msd_axes_t, time, ax):
    #msd_axes_t = analyze(traj, timestep)
    #print(t.shape, msd_self.shape)
    #average msd and linear fitting 
    msd_t = np.sum(msd_axes_t, axis=-1)
    mask1 = np.logical_and((time>25), (time<time[-1]-20))
    t_fit, msd_fit = linear_fit(time[mask1],msd_t[mask1])

    for msd_axis, m, c in zip(msd_axes_t.T, ['o', 's', '^'], ['tab:blue', 'tab:orange', 'tab:green']):
        ax.scatter(time, msd_axis,
                    marker=m,
                    linewidths = 0.5,
                    color=c,
                    edgecolor='slategrey')
    ax.scatter(time, msd_t,
                marker='P',
                linewidths = 0.5,
                color='tab:purple',
                edgecolor='slategrey')
    ax.plot(t_fit, msd_fit, '-', linewidth=2, color='k')
    ax.set_ylabel(r'MSD ($\AA{}^2$)')
    ax.set_xlabel('time (ps)')
    ax.legend([r'H$_x$',r'H$_y$',r'H$_z$', 'H', 'fit'], fontsize=8)
    #axes[1].set_yscale('log')
    #axes[1].set_xscale('log')

def plot_g_r(r_grid, g_r_pairs, atom_pairs, ax):
    for gr in g_r_pairs:
        gr /= np.sum(gr)
    leg = [i+'-'+j for (i,j) in atom_pairs]
    for gr in g_r_pairs:
        ax.plot(r_grid, gr)
    ax.legend(leg, fontsize=5)
    ax.set_ylabel(r'g(r) norm.')
    ax.set_xlabel(r'r ($\AA{}$)')
    ax.set_xlim([0,r_grid[-1] - 0.5])