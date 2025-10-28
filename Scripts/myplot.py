import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import time


def add_sizebar(ax, xlocs, ylocs, bcolor, text): # TODO:  add vertical and horizontal orientation
    """ Add a sizebar to the provided axis """
    ax.plot(xlocs, ylocs, ls='-', c=bcolor, linewidth=1., rasterized=True, clip_on=False)
    ax.text(x=xlocs[0]+10, y=ylocs[0]-2, s=text, va='center', ha='left', clip_on=False)


def plot_watermark(fig, **git_kwargs):
    """ Add simulation infomation on the figure """

    plt.text(.995, .99, '{0}\n {1} ({2}, {3})\n using "{4}"'.format(
        git_kwargs['timestamp'], git_kwargs['script_name'], git_kwargs['branch'], git_kwargs['short_hash'], git_kwargs['config_file']),
             transform=fig.transFigure, ha="right", va="top", clip_on=False,
             color = "black", family="Roboto Mono", weight="400", size="xx-small")
    

def plot_raster(t_spike_monitors: list, id_spike_monitors: list,
                 colors: list, cell_types: list[str], x_lim: list[float] = None, y_lim: list[float] = None,
                 stim_loc: list = None, stim_time: float = None, stim_dur: float = None, size_raster : float = 0.5,
                 **git_kwargs):
    
    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(12,7))

    # make raster plot
    # check if several spike_monitors or just one
    for i in range(len(t_spike_monitors)):
        ax.scatter(t_spike_monitors[i], id_spike_monitors[i], s=size_raster, marker='o', color=colors[i])

    # plot span for stimulation if stimulation
    if stim_time and stim_dur:
        ax.axvline(x=stim_time, color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)
        ax.axvline(x=stim_time + stim_dur, color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)
        ax.axvspan(stim_time, stim_time + stim_dur, alpha=.5, color=list(plt.cm.tab20c(19)[:3]), zorder=0)
        if stim_loc:
            for i in range(len(stim_loc)):
                ax.axhline(y=stim_loc[i], color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)
        trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
        ax.annotate('Stimulation', xy=(stim_time+stim_dur/2, 1.01 ), xycoords=trans, ha='center', color=list(plt.cm.tab20c(16)[:3]))
        
    # set axes
    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    # ax.axes.get_xaxis().set_visible(False)
    add_sizebar(ax, [x_lim[1]-250, x_lim[1]], [-1, -1], 'black', '250 ms')

    # set legend
    custom_lines = []

    for i in range(len(cell_types)):
        custom_lines.append(Line2D([0], [0], ls='', marker='o', color=colors[i], markersize=10, label=cell_types[i]))

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(custom_lines)
    ax.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0, 1.02, 1, 0.2), prop={'size':9})
    if git_kwargs:
        plot_watermark(fig, **git_kwargs)
    plt.show()


def save_raster(name_fig: str, t_spike_monitors: list, id_spike_monitors: list,
                 colors: list, cell_types: list[str], x_lim: list[float] = None, y_lim: list[float] = None,
                 stim_loc: list = None, stim_time: float = None, stim_dur: float = None, size_raster: float = 0.5,
                 **git_kwargs):

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(6,9))

    # make raster plot
    # check if several spike_monitors or just one
    for i in range(len(t_spike_monitors)):
        ax.scatter(t_spike_monitors[i], id_spike_monitors[i], s=size_raster, marker='o', color=colors[i])

    # plot span for stimulation if stimulation
    if stim_time and stim_dur:
        ax.axvline(x=stim_time, color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)
        ax.axvline(x=stim_time + stim_dur, color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)
        ax.axvspan(stim_time, stim_time + stim_dur, alpha=.5, color=list(plt.cm.tab20c(19)[:3]), zorder=0)
        if stim_loc:
            for i in range(len(stim_loc)):
                ax.axhline(y=stim_loc[i], color=list(plt.cm.tab20c(16)[:3]), ls='--', linewidth=1)
        trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
        ax.annotate('Stimulation', xy=(stim_time+stim_dur/2, 1.01 ), xycoords=trans, ha='center', color=list(plt.cm.tab20c(16)[:3]))

    # set axes
    if x_lim:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        ax.set_ylim(y_lim[0], y_lim[1])
    # ax.set_xlabel('Time (ms)')
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    add_sizebar(ax, [x_lim[1]-250, x_lim[1]], [-1, -1], 'black', '250 ms')

    # set legend
    custom_lines = []

    for i in range(len(cell_types)):
        custom_lines.append(Line2D([0], [0], ls='', marker='o', color=colors[i], markersize=10, label=cell_types[i]))

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(custom_lines)
    # ax.legend(handles=handles, ncol=1, loc='lower center', bbox_to_anchor=(1.2, 0.87), prop={'size':9})
    ax.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0, 1.02, 1, 0.2), prop={'size':9})
    if git_kwargs:
        plot_watermark(fig, **git_kwargs)
    plt.savefig(name_fig, bbox_inches="tight", pad_inches = 0)
    plt.clf()
    plt.close()


def plot_FR(t: list, rates: list, colors: list, cell_types: list[str], **git_kwargs):
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
    if git_kwargs:
        plot_watermark(fig, **git_kwargs)
    plt.show()


def save_FR(name_fig: str, t: list, rates: list, colors: list, cell_types: list[str], **git_kwargs):
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
    if git_kwargs:
        plot_watermark(fig, **git_kwargs)
    plt.savefig(name_fig, bbox_inches="tight", pad_inches = 0)
    plt.clf()
    plt.close()


def plot_specgram(t: list, f: list, sxx: list, cell_types: list[str], xlim: list=None, ylim: list=None, **git_kwargs): 

    imgs = [None] * len(t)
    cbars = [None] * len(t)
    fig, axs = plt.subplots(len(t), 1, figsize=(6, 9), sharex=True, sharey=True)
    for i in range(len(t)):
        # vmax = max(max(sxx[0].max(), sxx[1].max()), sxx[2].max())
        imgs[i] = axs[i].pcolormesh(t[i], f[i], sxx[i]/sxx[i].max(), shading='auto', cmap='inferno')

        axs[i].text(0.02, 0.9, cell_types[i], transform=axs[i].transAxes, color='white', verticalalignment='top')

        axs[i].axhline(y=30, xmin=0.0, xmax=5.0, color='white', linestyle='dashed', linewidth=2)

        axs[i].set_ylabel('Frequency (Hz)')

        cbars[i] = fig.colorbar(imgs[i], ax=axs[i])
        cbars[i].set_label("Power (normalized unit)")

    axs[-1].set_xlabel('Time (s)')

    if xlim:
        axs[-1].set_xlim(xlim)
    if ylim:
        axs[-1].set_ylim(ylim)
    else:
        axs[-1].set_ylim([0, 200])
        
    if git_kwargs:
        plot_watermark(fig, **git_kwargs)
    plt.show()


def save_specgram(name_fig: str, t: list, f: list, sxx: list, cell_types: list[str], xlim: list=None, ylim: list=None, **git_kwargs): 
    
    imgs = [None] * len(t)
    cbars = [None] * len(t)
    fig, axs = plt.subplots(len(t), 1, figsize=(6, 9), sharex=True, sharey=True)
    for i in range(len(t)):
        # vmax = max(max(sxx[0].max(), sxx[1].max()), sxx[2].max())
        imgs[i] = axs[i].pcolormesh(t[i], f[i], sxx[i]/sxx[i].max(), shading='auto', cmap='inferno')

        axs[i].text(0.02, 0.9, cell_types[i], transform=axs[i].transAxes, color='white', verticalalignment='top')

        axs[i].axhline(y=30, xmin=0.0, xmax=5.0, color='white', linestyle='dashed', linewidth=2)

        axs[i].set_ylabel('Frequency (Hz)')

        cbars[i] = fig.colorbar(imgs[i], ax=axs[i])
        cbars[i].set_label("Power (normalized unit)")

    axs[-1].set_xlabel('Time (s)')

    if xlim:
        axs[-1].set_xlim(xlim)
    if ylim:
        axs[-1].set_ylim(ylim)
    else:
        axs[-1].set_ylim([0, 200])

    if git_kwargs:
        plot_watermark(fig, **git_kwargs)
    plt.savefig(name_fig, bbox_inches="tight", pad_inches = 0)
    plt.clf()
    plt.close()


    
