import os
import sys
from pathlib import Path

fig2_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = Path(fig2_dir).parent
parent_dir = Path(scripts_dir).parent
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(parent_dir)
sys.path.append(scripts_dir)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

from utilities import *

def make_flat(old_list):
    ''' Takes as argument a list of lists and flattens it (i.e. returns a 1D list with all the elements) '''
    new_list = []
    for elem in old_list :
        if type(elem)==list:
            elem_flat = make_flat(elem)
            new_list += elem_flat
        else :
            new_list.append(elem)
    return new_list


def add_sizebar(ax, xlocs, ylocs, bcolor, text): # TODO:  add vertical and horizontal orientation
    """ Add a sizebar to the provided axis """
    ax.plot(xlocs, ylocs, ls='-', c=bcolor, linewidth=1., rasterized=True, clip_on=False)
    ax.text(x=xlocs[0]+10, y=ylocs[0]-2, s=text, va='center', ha='left', clip_on=False)


def plot_input(time_vec, amplitude_vec, axis):
    axis.plot(time_vec*1e-3, amplitude_vec, color='black')

    # remove spines
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    # show only first and last ticks for y values
    axis.set_yticks([amplitude_vec[0], amplitude_vec[-1]], labels=[0, 1], fontfamily='Arial', fontsize=13)

    # axis.set_xlim(left=0)
    axis.set_ylim(bottom=0, top=1)

    axis.set_title("Ramp input (nA)", fontfamily='Arial', fontsize=13)


def plot_specgram(axes, t_spikes_pyr, t_spikes_bc, t_spikes_olm, n_pyr_ca1, n_bc_ca1, n_olm_ca1, duration):
    # spectrogram
    cmesh_list = []
    vlow = []
    vhigh = []

    winsize_fr = 5 #ms
    overlap_fr = 0.9#0.9

    window_size = 1 #ms

    window_size_Pxx = 1000
    window_width_Pxx = int(window_size_Pxx * (1/window_size))
    overlap_Pxx = 0.9#0.9
    window_overlap_Pxx = int(window_width_Pxx*overlap_Pxx)
    vmin = 1e-12
    vmax = 1.
    norm_bc = colors.Normalize(vmin=vmin, vmax=vmax)
    norm_olm = colors.Normalize(vmin=vmin, vmax=vmax)
    norm_pyr = colors.Normalize(vmin=vmin, vmax=vmax)

    specgram_kwargs = { 'return_onesided' : True,
                        'scaling' : 'density',
                        'mode' : 'magnitude' }

    # compute firing rates
    t_FR_pyr, count_pyr, FR_pyr, fs_n = compute_FR(np.array(t_spikes_pyr)*1e-3, n_pyr_ca1, duration*1e-3, winsize_fr*1e-3, overlap_fr)
    t_FR_bc, count_bc, FR_bc, _ = compute_FR(np.array(t_spikes_bc)*1e-3, n_bc_ca1, duration*1e-3, winsize_fr*1e-3, overlap_fr)
    t_FR_olm, count_olm, FR_olm, _ = compute_FR(np.array(t_spikes_olm)*1e-3, n_olm_ca1, duration*1e-3, winsize_fr*1e-3, overlap_fr)

    fv_pyr, tv_pyr, pspec_pyr = my_specgram2(FR_pyr, fs_n, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
    fv_bc, tv_bc, pspec_bc = my_specgram2(FR_bc, fs_n, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
    fv_olm, tv_olm, pspec_olm = my_specgram2(FR_olm, fs_n, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)

    vlow.append(pspec_pyr.min())
    vhigh.append(pspec_pyr.max())

    for cmsh in make_flat(cmesh_list):
        cmsh.set_clim(min(vlow), max(vhigh))

    vmax = max(max(pspec_pyr.max(), pspec_bc.max()), pspec_olm.max())
    im_pyr = axes[0].pcolormesh(tv_pyr, fv_pyr, pspec_pyr/vmax, shading='auto', cmap='inferno')
    im_bc = axes[1].pcolormesh(tv_bc, fv_bc, pspec_bc/vmax, shading='auto', cmap='inferno')
    im_olm = axes[2].pcolormesh(tv_olm, fv_olm, pspec_olm/vmax, shading='auto', cmap='inferno')

    # axes[0].sharex(axes[1])
    # axes[1].sharex(axes[2])

    axes[0].sharey(axes[1])
    axes[1].sharey(axes[2])

    axes[0].set_ylim(bottom=0, top=100)
    axes[1].set_ylim(bottom=0, top=100)
    axes[2].set_ylim(bottom=0, top=100)

    axes[0].set_xlim(left=0.22, right=4.72)
    axes[1].set_xlim(left=0.22)
    axes[2].set_xlim(left=0.22)

    axes[0].set_xticks([1, 2, 3, 4], labels=[1, 2, 3, 4], color="w")
    axes[1].set_xticks([1, 2, 3, 4], labels=[1, 2, 3, 4], color="w")

    axes[0].set_yticks([25, 50, 100], labels=[25, 50, 100], fontfamily="Arial", fontsize=13)
    axes[1].set_yticks([25, 50, 100], labels=[25, 50, 100], fontfamily="Arial", fontsize=13)
    axes[2].set_yticks([25, 50, 100], labels=[25, 50, 100], fontfamily="Arial", fontsize=13)

    axes[0].set_title("Pyramidal cells", fontfamily="Arial", fontsize=13)
    axes[1].set_title("Basket cells", fontfamily="Arial", fontsize=13)
    axes[2].set_title("OLM cells", fontfamily="Arial", fontsize=13)

    # cbar1 = fig.colorbar(im_pyr, ax=axes[0], aspect=5)
    # cbar2 = fig.colorbar(im_bc, ax=axes[1], aspect=5)
    # cbar3 = fig.colorbar(im_olm, ax=axes[2], aspect=5)
    # cbar1.ax.tick_params(labelsize=13)
    # cbar2.ax.tick_params(labelsize=13)
    # cbar3.ax.tick_params(labelsize=13)
    # cbar1.set_label("Power (arbitrary unit)")
    # cbar2.set_label("Power (arbitrary unit)")
    # cbar3.set_label("Power (arbitrary unit)")

    return axes[0].get_xlim()


def plot_raster(axis, t_spike_monitors: list, id_spike_monitors: list, colors: list, 
                x_lim: list[float] = None, y_lim: list[float] = None, size_raster : float = 0.1):
    
    # make raster plot
    # check if several spike_monitors or just one
    for i in range(len(t_spike_monitors)):
        axis.scatter(t_spike_monitors[i]*1e-3, id_spike_monitors[i], s=size_raster, marker='o', color=colors[i])

    # set axes
    if x_lim:
        axis.set_xlim(x_lim[0], x_lim[1])
    if y_lim:
        axis.set_ylim(y_lim[0], y_lim[1])

    # remove spines
    axis.spines['left'].set_visible(False)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    axis.axes.get_yaxis().set_visible(False)
    # ax.axes.get_xaxis().set_visible(False)
    # add_sizebar(axis, [5000-250, 5000], [-1, -1], 'black', '250 ms')
    axis.set_xticks([1, 2, 3, 4], labels=[1, 2, 3, 4], fontfamily="Arial", fontsize=13)
    # axis.set_xlim(left=0)
    axis.set_xlabel("Time (s)", fontfamily="Arial", fontsize=13)



if __name__ == "__main__":
    # set saving directory
    result_dir = os.path.join(parent_dir, "new_param_search_paper")

    # retrieve data dir
    weight_E = 2.0
    weight_I = 0.1
    data_dir = os.path.join(result_dir, f"w_E_{weight_E}", f"w_I_{weight_I}", "data")

    # load data
    ## input
    ramp_input = np.load(os.path.join(data_dir, "ramping_current.npz"))

    ## spike monitors
    spikemon_pyr = np.load(os.path.join(data_dir, "CA1_pyr_spikemon.npz"))
    spikemon_bc = np.load(os.path.join(data_dir, "CA1_bc_spikemon.npz"))
    spikemon_olm = np.load(os.path.join(data_dir, "CA1_olm_spikemon.npz"))

    # start fig2

    fig, axs = plt.subplots(5, 1, figsize=(10, 5), sharex=False, height_ratios=[1/10, 2/10, 2/10, 2/10, 3/10])

    ## input
    plot_input(ramp_input['t'], ramp_input['amplitude'], axs[0])

    ## rasters
    # get spike times and associated cell ids
    spkt_pyr = spikemon_pyr['t_spike']
    spkid_pyr = spikemon_pyr['cell_id']

    spkt_bc = spikemon_bc['t_spike']
    spkid_bc = spikemon_bc['cell_id']

    spkt_olm = spikemon_olm['t_spike']
    spkid_olm = spikemon_olm['cell_id']
    
    # pick 50 random pyramidal cells
    # pyr_ids = random.sample(range(100), 50)
    pyr_ids = [i for i in range(50)]

    indices = [i for i, cid in enumerate(spkid_pyr) if cid in pyr_ids]
    spkt_pyr_filtered = spkt_pyr[indices]
    spkid_pyr_filtered = spkid_pyr[indices]

    plot_raster(axs[4], [spkt_pyr_filtered, spkt_bc, spkt_olm], [spkid_pyr_filtered, spkid_bc+2-50, spkid_olm+4-50], ['C0', 'C3', 'C1'])

    ## spectrograms
    xlim = plot_specgram([axs[1], axs[2], axs[3]], spkt_pyr, spkt_bc, spkt_olm, 100, 9, 3, 5000)

    # Adjust spacing
    plt.subplots_adjust(hspace=0.5)
    plt.show()

