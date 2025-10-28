import os
import sys

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from myplot import *
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


if __name__ == "__main__":
    ### set data directory name
    w_sca = 1.0
    offset = 50
    data_dir_name = f"2025_10_22 16H03M21 no stim - noisy input offset_{offset} wsca_{w_sca}"
    # data_dir_name = f"SCA_weights/0.02nA/w_ScaE_0.5"
    script_dir = os.path.dirname(os.path.abspath(__file__)) # this dis
    parent_dir = Path(script_dir).parent # main dir

    results_dir = "network_with_schaffers_results"
    # results_dir = "new_param_search_paper_good_OLM"
    # weight = 0.3

    data_dir = os.path.join(parent_dir, results_dir, data_dir_name, "data")

    # for watermaks on figures -> reproducibility
    git_kwargs = {'timestamp': time.ctime(),
                'branch': get_git_revision_branch(), 
                'hash': get_git_revision_hash(),
                'script_name': os.path.basename(__file__)}

    # conn_mat = np.load(os.path.join(data_dir, "connection_matrix.npy"))
    # print(conn_mat)

    # sys.exit()

    spikes_pyr = np.load(os.path.join(data_dir, "CA1_pyr_spikemon.npz"))
    t_spikes_pyr = spikes_pyr['t_spike']
    cell_id_pyr = spikes_pyr['cell_id']

    # id = np.argwhere((spikes_pyr['t_spike'] >= 3000) & (spikes_pyr['t_spike'] <= 4000)).reshape(-1,)
    # t_spikes_pyr = spikes_pyr['t_spike'][id]
    # cell_id = spikes_pyr['cell_id'][id]
    # cell_ids = np.unique(cell_id)

    # for t, cell in zip(t_spikes_pyr, cell_id):
    #     if cell == 19:
    #         print(t)
    # # print(cell_ids)
    # # print(cell_id)

    # sys.exit()
    # # t_spikes_pyr = spikes_pyr['t_spike'][id]
    # cell_id = spikes_pyr['cell_id'][id]
    # cell_ids = np.unique(cell_id)
    # print(cell_ids)
    # sys.exit()
    # t_spikes_pyr = []
    # cell_id_pyr = []
    # for t, cell in zip(spikes_pyr['t_spike'], spikes_pyr['cell_id']):
    #     if cell in cell_ids:
    #         t_spikes_pyr.append(t)
    #         cell_id_pyr.append(cell)

    # t_spikes_pyr = np.array(t_spikes_pyr)
    # cell_id_pyr = np.array(cell_id_pyr)

    spikes_bc = np.load(os.path.join(data_dir, "CA1_bc_spikemon.npz"))
    t_spikes_bc = spikes_bc['t_spike']
    cell_id_bc = spikes_bc['cell_id']

    spikes_olm = np.load(os.path.join(data_dir, "CA1_olm_spikemon.npz"))
    t_spikes_olm = spikes_olm['t_spike']
    cell_id_olm = spikes_olm['cell_id']

    spikes_sca = np.load(os.path.join(data_dir, "CA3_sca_last_spikemon.npz"))
    t_spikes_sca = spikes_sca['t_spike']
    cell_id_sca = spikes_sca['cell_id']

    # set parameters
    cmesh_list = []
    vlow = []
    vhigh = []

    N_pyr = 100 #len(np.unique(cell_id_pyr)) # 100
    N_bc = 9 #len(np.unique(cell_id_bc))
    N_olm = 3 #len(np.unique(cell_id_olm))
    N_sca = 26

    dur = 5000 #ms
    winsize_fr = 5 #ms
    overlap_fr = 0.9

    window_size = 1 #ms

    window_size_Pxx = 150 # for theta 6Hz
    window_width_Pxx = int(window_size_Pxx * (1/window_size))
    overlap_Pxx = 0.9
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
    t_FR_pyr, count_pyr, FR_pyr, fs_n_pyr = compute_FR(t_spikes_pyr*1e-3, N_pyr, dur*1e-3, winsize_fr*1e-3, overlap_fr)
    t_FR_bc, count_bc, FR_bc, fs_n_bc = compute_FR(t_spikes_bc*1e-3, N_bc, dur*1e-3, winsize_fr*1e-3, overlap_fr)
    t_FR_olm, count_olm, FR_olm, fs_n_olm = compute_FR(t_spikes_olm*1e-3, N_olm, dur*1e-3, winsize_fr*1e-3, overlap_fr)
    t_FR_sca, count_sca, FR_sca, fs_n_sca = compute_FR(t_spikes_sca*1e-3, N_sca, dur*1e-3, winsize_fr*1e-3, overlap_fr)

    # compute spectrograms
    # fv_pyr, tv_pyr, pspec_pyr = my_specgram(FR_pyr, fs_n, window_width_Pxx, window_overlap_Pxx)
    # fv_bc, tv_bc, pspec_bc = my_specgram(FR_bc, fs_n, window_width_Pxx, window_overlap_Pxx)
    # fv_olm, tv_olm, pspec_olm = my_specgram(FR_olm, fs_n, window_width_Pxx, window_overlap_Pxx)

    fv_pyr, tv_pyr, pspec_pyr = my_specgram2(FR_pyr, fs_n_pyr, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
    fv_bc, tv_bc, pspec_bc = my_specgram2(FR_bc, fs_n_bc, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
    fv_olm, tv_olm, pspec_olm = my_specgram2(FR_olm, fs_n_olm, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
    fv_sca, tv_sca, pspec_sca = my_specgram2(FR_sca, fs_n_sca, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)

    vlow.append(pspec_pyr.min())
    vhigh.append(pspec_pyr.max())

    # fig, ax = plt.subplots(1,1, figsize=(8, 7))

    # cmesh = ax.pcolormesh(tv_pyr, fv_pyr, pspec_pyr, cmap='inferno', shading='auto', rasterized=True)
    # cmesh_list.append(cmesh)

    # plot_specgram([tv_pyr, tv_bc, tv_olm, tv_sca], [fv_pyr, fv_bc, fv_olm, fv_sca], [pspec_pyr, pspec_bc, pspec_olm, pspec_sca], ["pyramidal cells", "basket cells", "olm cells", "schaffer collaterals"], ylim=[0,125], xlim=[3,4])
    save_specgram(os.path.join(parent_dir, results_dir, data_dir_name, "figures", "test_specgram_1s_with_schaffer.png"), [tv_pyr, tv_bc, tv_olm, tv_sca], [fv_pyr, fv_bc, fv_olm, fv_sca], [pspec_pyr, pspec_bc, pspec_olm, pspec_sca], ["pyramidal cells", "basket cells", "olm cells", "schaffer_collaterals"], ylim=[0,100], xlim=[3,4], **git_kwargs)
    for cmsh in make_flat(cmesh_list):
        cmsh.set_clim(min(vlow), max(vhigh))

    # for ax,cmsh in zip(make_flat(ax_cbars), make_flat(cmesh_list)):
    #         cbar = fig.colorbar(cmsh, cax=ax)
    #         cbar.outline.set_color('black')
    #         cbar.outline.set_linewidth(0.5)
    #         cbar.solids.set_rasterized(True)
    #         cbar.dividers.set_color('none')
    #         cbar.dividers.set_linewidth(5)
    #         cbar.ax.tick_params(labelsize=fsize_ticks)