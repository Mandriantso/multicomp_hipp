import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def add_sizebar(ax, xlocs, ylocs, bcolor, text):
    """ Add a sizebar to the provided axis """
    ax.plot(xlocs, ylocs, ls='-', c=bcolor, linewidth=1., rasterized=True, clip_on=False)
    ax.text(x=xlocs[0]+10, y=ylocs[0], s=text, va='center', ha='left', clip_on=False)


def plot_raster(t_spike_monitors: list, id_spike_monitors: list,
                 colors: list, cell_types: list[str], x_lim: list[float] = None, y_lim: list[float] = None,
                 stim_loc: float = None, stim_time: float = None, stim_dur: float = None, size_raster : float = 0.5):
    
    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(6,9))

    # make raster plot
    # check if several spike_monitors or just one
    for i in range(len(t_spike_monitors)):
        ax.scatter(t_spike_monitors[i], id_spike_monitors[i], s=size_raster, marker='o', color=colors[i])

    # plot span for stimulation if stimulation
    if stim_loc:
        ax.axvline(x=stim_time, color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)
        ax.axvline(x=stim_time + stim_dur, color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)
        ax.axvspan(stim_time, stim_time + stim_dur, alpha=.5, color=list(plt.cm.tab20c(19)[:3]), zorder=0)
        ax.axhline(y=stim_loc, color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)

        trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
        ax.annotate('Stimulation', xy=(stim_time+stim_dur/2, 1.01 ), xycoords=trans, ha='center', color=list(plt.cm.tab20c(16)[:3]))

    # set axes
    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Cell n°')

    # set legend
    custom_lines = []

    for i in range(len(cell_types)):
        custom_lines.append(Line2D([0], [0], ls='', marker='o', color=colors[i], markersize=10, label=cell_types[i]))

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(custom_lines)
    ax.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0, 1.02, 1, 0.2), prop={'size':9})

    plt.show()


def save_raster(name_fig: str, t_spike_monitors: list, id_spike_monitors: list,
                 colors: list, cell_types: list[str], x_lim: list[float] = None, y_lim: list[float] = None,
                 stim_loc: float = None, stim_time: float = None, stim_dur: float = None, size_raster: float = 0.5):

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(6,9))

    # make raster plot
    # check if several spike_monitors or just one
    for i in range(len(t_spike_monitors)):
        ax.scatter(t_spike_monitors[i], id_spike_monitors[i], s=size_raster, marker='o', color=colors[i])

    # plot span for stimulation if stimulation
    if stim_loc:
        ax.axvline(x=stim_time, color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)
        ax.axvline(x=stim_time + stim_dur, color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)
        ax.axvspan(stim_time, stim_time + stim_dur, alpha=.5, color=list(plt.cm.tab20c(19)[:3]), zorder=0)
        ax.axhline(y=stim_loc, color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)

        trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
        ax.annotate('Stimulation', xy=(stim_time+stim_dur/2, 1.01 ), xycoords=trans, ha='center', color=list(plt.cm.tab20c(16)[:3]))

    # set axes
    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Cell n°')

    # set legend
    custom_lines = []

    for i in range(len(cell_types)):
        custom_lines.append(Line2D([0], [0], ls='', marker='o', color=colors[i], markersize=10, label=cell_types[i]))

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(custom_lines)
    # ax.legend(handles=handles, ncol=1, loc='lower center', bbox_to_anchor=(1.2, 0.87), prop={'size':9})
    ax.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0, 1.02, 1, 0.2), prop={'size':9})

    plt.savefig(name_fig, bbox_inches="tight")
    plt.close()


def plot_FR(t: list, rates: list, colors: list, cell_types: list[str]):
    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(6,9))
    k=110
    for i in range(len(rates)):
        ax.plot(t[i], rates[i]- i*k, color=colors[i])

    ax.set_xlabel('Time (ms)')
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    add_sizebar(ax, [t[0][-1]]*2, [0-len(rates)*k, 30-len(rates)*k], 'black', '30 Hz')

    # set legend
    custom_lines = []

    for i in range(len(cell_types)):
        custom_lines.append(Line2D([0], [0], ls='', marker='_', color=colors[i], markersize=10, label=cell_types[i]))

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(custom_lines)
    ax.legend(handles=handles, ncol=1, loc='lower center', bbox_to_anchor=(1.2, 0.87), prop={'size':9})

    plt.show()


def save_FR(name_fig: str, t: list, rates: list, colors: list, cell_types: list[str]):
    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(6,9))
    k=110
    for i in range(len(rates)):
        ax.plot(t[i], rates[i] - 0.4*i*k, color=colors[i])

    ax.set_xlabel('Time (ms)')
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    print([t[0][-1]]*2)
    add_sizebar(ax, [t[0][-1]]*2, [0-len(rates)*k, 30-len(rates)*k], 'black', '30 Hz')

    # set legend
    custom_lines = []

    for i in range(len(cell_types)):
        custom_lines.append(Line2D([0], [0], ls='', marker='_', color=colors[i], markersize=10, label=cell_types[i]))

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(custom_lines)
    ax.legend(handles=handles, ncol=1, loc='lower center', bbox_to_anchor=(1.2, 0.87), prop={'size':9})

    plt.savefig(name_fig, bbox_inches="tight")
    plt.close()


def plot_specgram(t: list, f: list, sxx: list, cell_types: list[str], ylim: list=None): 

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 9), sharex=True, sharey=True)
    vmax = max(max(sxx[0].max(), sxx[1].max()), sxx[2].max())
    im_pyr = ax1.pcolormesh(t[0], f[0], sxx[0]/vmax, shading='auto', cmap='inferno')
    im_bc = ax2.pcolormesh(t[1], f[1], sxx[1]/vmax, shading='auto', cmap='inferno')
    im_olm = ax3.pcolormesh(t[2], f[2], sxx[2]/vmax, shading='auto', cmap='inferno')

    ax1.text(0.02, 0.9, cell_types[0], transform=ax1.transAxes, color='white', verticalalignment='top')
    ax2.text(0.02, 0.9, cell_types[1], transform=ax2.transAxes, color='white', verticalalignment='top')
    ax3.text(0.02, 0.9, cell_types[2], transform=ax3.transAxes, color='white', verticalalignment='top')

    ax3.set_xlabel('Time [s]')
    if ylim:
        ax3.set_ylim(ylim)
    else:
        ax3.set_ylim([0,200])
    cbar1 = fig.colorbar(im_pyr, ax=ax1)
    cbar2 = fig.colorbar(im_bc, ax=ax2)
    cbar3 = fig.colorbar(im_olm, ax=ax3)

    plt.show()


def save_specgram(name_fig: str, t: list, f: list, sxx: list, cell_types: list[str], ylim: list=None): 

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 9), sharex=True, sharey=True)
    vmax = max(max(sxx[0].max(), sxx[1].max()), sxx[2].max())
    im_pyr = ax1.pcolormesh(t[0], f[0], sxx[0]/vmax, shading='auto', cmap='inferno')
    im_bc = ax2.pcolormesh(t[1], f[1], sxx[1]/vmax, shading='auto', cmap='inferno')
    im_olm = ax3.pcolormesh(t[2], f[2], sxx[2]/vmax, shading='auto', cmap='inferno')

    ax1.text(0.02, 0.9, cell_types[0], transform=ax1.transAxes, color='white', verticalalignment='top')
    ax2.text(0.02, 0.9, cell_types[1], transform=ax2.transAxes, color='white', verticalalignment='top')
    ax3.text(0.02, 0.9, cell_types[2], transform=ax3.transAxes, color='white', verticalalignment='top')

    ax3.set_xlabel('Time [s]')
    if ylim:
        ax3.set_ylim(ylim)
    else:
        ax3.set_ylim([0,200])

    cbar1 = fig.colorbar(im_pyr, ax=ax1)
    cbar2 = fig.colorbar(im_bc, ax=ax2)
    cbar3 = fig.colorbar(im_olm, ax=ax3)

    plt.savefig(name_fig, bbox_inches="tight")
    plt.close()


    
