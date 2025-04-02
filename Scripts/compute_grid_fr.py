import os
import sys

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from myplot import *
from utilities import *

# retrieve data
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = Path(script_dir).parent
result_dir = os.path.join(parent_dir, "new_param_search")

def main():
    k_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.] 
    k_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.] 

    # mean_FR = np.array((len(k_factors), len(k_factors)))
    # mean_FR_instantaneous = np.array((len(k_factors), len(k_factors)))

    mean_FR_pyr_pop = np.zeros((len(k_x), len(k_factors)))
    # mean_FR_pyr_ind = np.array((len(k_factors), len(k_factors)))
    mean_FR_pyr_instantaneous = np.zeros((len(k_x), len(k_factors)))

    mean_FR_bc_pop = np.zeros((len(k_x), len(k_factors)))
    # mean_FR_bc_ind = np.array((len(k_factors), len(k_factors)))
    mean_FR_bc_instantaneous = np.zeros((len(k_x), len(k_factors)))

    mean_FR_olm_pop = np.zeros((len(k_x), len(k_factors)))
    # mean_FR_olm_ind = np.array((len(k_factors), len(k_factors)))
    mean_FR_olm_instantaneous = np.zeros((len(k_x), len(k_factors)))



    for i, w_E in enumerate(k_x):
        w_E_dir = os.path.join(result_dir, "w_E_{} netstim".format(w_E))
        for j, w_I in enumerate(k_factors):
            w_I_dir = os.path.join(w_E_dir, "w_I_{}".format(w_I))
            data_dir = os.path.join(w_I_dir, "data")

            spikes_pyr = np.load(os.path.join(data_dir, "CA1_pyr_spikemon.npz"))
            t_spikes_pyr = spikes_pyr['t_spike']
            cell_id_pyr = spikes_pyr['cell_id']

            spikes_bc = np.load(os.path.join(data_dir, "CA1_bc_spikemon.npz"))
            t_spikes_bc = spikes_bc['t_spike']
            cell_id_bc = spikes_bc['cell_id']

            spikes_olm = np.load(os.path.join(data_dir, "CA1_olm_spikemon.npz"))
            t_spikes_olm = spikes_olm['t_spike']
            cell_id_olm = spikes_olm['cell_id']

            # populations firing rate
            pyr_last_sec = []
            for t, id in zip(t_spikes_pyr, cell_id_pyr):
                if t >= 4000: # TODO: retrieve TSTOP and compute TSTOP - 1e3s
                    pyr_last_sec.append((id, t))

            pyr_last_sec = sorted(pyr_last_sec, key=lambda element: element[0])
            pyr_last_sec = np.array(pyr_last_sec)

            bc_last_sec = []
            for t, id in zip(t_spikes_bc, cell_id_bc):
                if t >= 4000: # TODO: retrieve TSTOP and compute TSTOP - 1e3s
                    bc_last_sec.append((id, t))

            bc_last_sec = sorted(bc_last_sec, key=lambda element: element[0])
            bc_last_sec  = np.array(bc_last_sec)

            olm_last_sec = []
            for t, id in zip(t_spikes_olm, cell_id_olm):
                if t >= 4000: # TODO: retrieve TSTOP and compute TSTOP - 1e3s
                    olm_last_sec.append((id, t))

            olm_last_sec = sorted(olm_last_sec, key=lambda element: element[0])
            olm_last_sec = np.array(olm_last_sec)

            print("spikes bc : {}".format(t_spikes_bc[t_spikes_bc>=4000]))
            print("bc_last_sec : {}".format([el[1] for el in bc_last_sec if el[0]==100]))
            print(bc_last_sec[:,1]*1e-3)
            print(len(pyr_last_sec[:,1]))
            print(len(pyr_last_sec[:,1])/100)
            # print("spikes bc last sec id 0 : {}".format(bc_last_sec[bc_last_sec[]]))
            # sys.exit()

            # set parameters
            cmesh_list = []
            vlow = []
            vhigh = []

            N_pyr = 100 #len(np.unique(cell_id_pyr)) # 100
            N_bc = 9 #len(np.unique(cell_id_bc))
            N_olm = 3 #len(np.unique(cell_id_olm))

            dur = 1000 #ms
            winsize_fr = 5 #ms
            overlap_fr = 0.9

            window_size = 1 #ms

            window_size_Pxx = 1000
            window_width_Pxx = int(window_size_Pxx * (1/window_size))
            overlap_Pxx = 0.9
            window_overlap_Pxx = int(window_width_Pxx*overlap_Pxx)

            specgram_kwargs = { 'return_onesided' : True,
                        'scaling' : 'density',
                        'mode' : 'magnitude' }

            # compute firing rates
            t_FR_pyr, count_pyr, FR_pyr, fs_n_pyr = compute_FR((pyr_last_sec[:,1]-4000)*1e-3, N_pyr, dur*1e-3, winsize_fr*1e-3, overlap_fr)
            t_FR_bc, count_bc, FR_bc, fs_n_bc = compute_FR((bc_last_sec[:,1]-4000)*1e-3, N_bc, dur*1e-3, winsize_fr*1e-3, overlap_fr)
            t_FR_olm, count_olm, FR_olm, fs_n_olm = compute_FR((olm_last_sec[:,1]-4000)*1e-3, N_olm, dur*1e-3, winsize_fr*1e-3, overlap_fr)

            print("FR_pyr : {}".format(np.mean(FR_pyr)))

            mean_FR_pyr_pop[i,j] = np.mean(FR_pyr)
            mean_FR_bc_pop[i,j] = np.mean(FR_bc)
            mean_FR_olm_pop[i,j] = np.mean(FR_olm)

            # compute spectrograms
            fv_pyr, tv_pyr, pspec_pyr = my_specgram2(FR_pyr, fs_n_pyr, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
            fv_bc, tv_bc, pspec_bc = my_specgram2(FR_bc, fs_n_bc, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
            fv_olm, tv_olm, pspec_olm = my_specgram2(FR_olm, fs_n_olm, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)

            ifr_pyr = np.sum(pspec_pyr[:100, :], axis=0)  # Sum the power in the low-frequency band (0-100 Hz)
            ifr_bc = np.sum(pspec_bc[:100, :], axis=0)
            ifr_olm = np.sum(pspec_olm[:100, :], axis=0)

            # Convert the instantaneous firing rate to Hz (firing rate per second)
            ifr_pyr = ifr_pyr * (1000 / window_width_Pxx)  # scale to Hz
            ifr_bc = ifr_bc * (1000 / window_width_Pxx)
            ifr_olm = ifr_olm * (1000 / window_width_Pxx)

            print("FR_pyr_instant spec : {}".format(np.mean(ifr_pyr)))

            mean_FR_pyr_instantaneous[i,j] = np.mean(ifr_pyr)
            mean_FR_bc_instantaneous[i,j] = np.mean(ifr_bc)
            mean_FR_olm_instantaneous[i,j] = np.mean(ifr_olm)

    # save vectors
    np.save(os.path.join(result_dir, 'mean_FR_pyr_full.npy'), mean_FR_pyr_pop)
    np.save(os.path.join(result_dir, 'mean_FR_bc_full.npy'), mean_FR_bc_pop)
    np.save(os.path.join(result_dir, 'mean_FR_olm_full.npy'), mean_FR_olm_pop)

    np.save(os.path.join(result_dir, 'mean_iFR_pyr_full.npy'), mean_FR_pyr_instantaneous)
    np.save(os.path.join(result_dir, 'mean_iFR_bc_full.npy'), mean_FR_bc_instantaneous)
    np.save(os.path.join(result_dir, 'mean_iFR_olm_full.npy'), mean_FR_olm_instantaneous)

def compute_iFR():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = Path(script_dir).parent
    result_dir = os.path.join(parent_dir, "new_param_search")

    k_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.] 

    mean_FR_pyr_instantaneous = np.zeros((len(k_factors), len(k_factors)))

    mean_FR_bc_instantaneous = np.zeros((len(k_factors), len(k_factors)))

    mean_FR_olm_instantaneous = np.zeros((len(k_factors), len(k_factors)))



    for i, w_E in enumerate(k_factors):
        w_E_dir = os.path.join(result_dir, "w_E_{} netstim".format(w_E))
        for j, w_I in enumerate(k_factors):
            w_I_dir = os.path.join(w_E_dir, "w_I_{}".format(w_I))
            data_dir = os.path.join(w_I_dir, "data")

            spikes_pyr = np.load(os.path.join(data_dir, "CA1_pyr_spikemon.npz"))
            t_spikes_pyr = spikes_pyr['t_spike']
            cell_id_pyr = spikes_pyr['cell_id']

            spikes_bc = np.load(os.path.join(data_dir, "CA1_bc_spikemon.npz"))
            t_spikes_bc = spikes_bc['t_spike']
            cell_id_bc = spikes_bc['cell_id']

            spikes_olm = np.load(os.path.join(data_dir, "CA1_olm_spikemon.npz"))
            t_spikes_olm = spikes_olm['t_spike']
            cell_id_olm = spikes_olm['cell_id']

            # populations firing rate
            pyr_last_sec = []
            for t, id in zip(t_spikes_pyr, cell_id_pyr):
                if t >= 4000: # TODO: retrieve TSTOP and compute TSTOP - 1e3s
                    pyr_last_sec.append((id, t))

            pyr_last_sec = sorted(pyr_last_sec, key=lambda element: element[0])
            pyr_last_sec = np.array(pyr_last_sec)

            bc_last_sec = []
            for t, id in zip(t_spikes_bc, cell_id_bc):
                if t >= 4000: # TODO: retrieve TSTOP and compute TSTOP - 1e3s
                    bc_last_sec.append((id, t))

            bc_last_sec = sorted(bc_last_sec, key=lambda element: element[0])
            bc_last_sec  = np.array(bc_last_sec)

            olm_last_sec = []
            for t, id in zip(t_spikes_olm, cell_id_olm):
                if t >= 4000: # TODO: retrieve TSTOP and compute TSTOP - 1e3s
                    olm_last_sec.append((id, t))

            olm_last_sec = sorted(olm_last_sec, key=lambda element: element[0])
            olm_last_sec = np.array(olm_last_sec)

            # set parameters
            cmesh_list = []
            vlow = []
            vhigh = []

            N_pyr = 100 #len(np.unique(cell_id_pyr)) # 100
            N_bc = 9 #len(np.unique(cell_id_bc))
            N_olm = 3 #len(np.unique(cell_id_olm))

            dur = 1000 #ms
            winsize_fr = 5 #ms
            overlap_fr = 0.9

            window_size = 1 #ms

            window_size_Pxx = 1000
            window_width_Pxx = int(window_size_Pxx * (1/window_size))
            overlap_Pxx = 0.9
            window_overlap_Pxx = int(window_width_Pxx*overlap_Pxx)

            specgram_kwargs = { 'return_onesided' : True,
                        'scaling' : 'density',
                        'mode' : 'magnitude' }

            # compute firing rates
            t_FR_pyr, count_pyr, FR_pyr, fs_n_pyr = compute_FR((pyr_last_sec[:,1]-4000)*1e-3, N_pyr, dur*1e-3, winsize_fr*1e-3, overlap_fr)
            t_FR_bc, count_bc, FR_bc, fs_n_bc = compute_FR((bc_last_sec[:,1]-4000)*1e-3, N_bc, dur*1e-3, winsize_fr*1e-3, overlap_fr)
            t_FR_olm, count_olm, FR_olm, fs_n_olm = compute_FR((olm_last_sec[:,1]-4000)*1e-3, N_olm, dur*1e-3, winsize_fr*1e-3, overlap_fr)

            # compute spectrograms
            fv_pyr, tv_pyr, pspec_pyr = my_specgram2(FR_pyr, fs_n_pyr, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
            fv_bc, tv_bc, pspec_bc = my_specgram2(FR_bc, fs_n_bc, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
            fv_olm, tv_olm, pspec_olm = my_specgram2(FR_olm, fs_n_olm, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)

            print("pspec_pyr shape : {}".format(np.shape(pspec_pyr)))
            print("pspec_bc shape : {}".format(np.shape(pspec_bc)))
            print("pspec_olm shape : {}".format(np.shape(pspec_olm)))
            print("x axis : {}".format(pspec_pyr[:]))
            print("y axis : {}".format(pspec_pyr[:,1]))
            sys.exit()
            ifr_pyr = np.sum(pspec_pyr[:100, :], axis=0)  # Sum the power in the low-frequency band (0-100 Hz)
            ifr_bc = np.sum(pspec_bc[:100, :], axis=0)
            ifr_olm = np.sum(pspec_olm[:100, :], axis=0)

            # Convert the instantaneous firing rate to Hz (firing rate per second)
            ifr_pyr = ifr_pyr * (1000 / window_width_Pxx)  # scale to Hz
            ifr_bc = ifr_bc * (1000 / window_width_Pxx)
            ifr_olm = ifr_olm * (1000 / window_width_Pxx)

            print("FR_pyr_instant spec : {}".format(np.mean(ifr_pyr)))

            mean_FR_pyr_instantaneous[i,j] = np.mean(ifr_pyr)
            mean_FR_bc_instantaneous[i,j] = np.mean(ifr_bc)
            mean_FR_olm_instantaneous[i,j] = np.mean(ifr_olm)

    # save vectors
    np.save(os.path.join(result_dir, 'mean_FR_pyr.npy'), mean_FR_pyr_pop)
    np.save(os.path.join(result_dir, 'mean_FR_bc.npy'), mean_FR_bc_pop)
    np.save(os.path.join(result_dir, 'mean_FR_olm.npy'), mean_FR_olm_pop)

    np.save(os.path.join(result_dir, 'mean_iFR_pyr.npy'), mean_FR_pyr_instantaneous)
    np.save(os.path.join(result_dir, 'mean_iFR_bc.npy'), mean_FR_bc_instantaneous)
    np.save(os.path.join(result_dir, 'mean_iFR_olm.npy'), mean_FR_olm_instantaneous)

    


if __name__=="__main__":

    # compute_iFR()
    # sys.exit()
    # retrieve data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = Path(script_dir).parent
    result_dir = os.path.join(parent_dir, "new_param_search")
    print("data loaded")
    k_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.] 
    k_x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.] 
    main()

    mean_FR_pyr_pop = np.load(os.path.join(result_dir, 'mean_FR_pyr_full.npy'))
    mean_FR_bc_pop = np.load(os.path.join(result_dir, 'mean_FR_bc_full.npy'))
    mean_FR_olm_pop = np.load(os.path.join(result_dir, 'mean_FR_olm_full.npy'))

    mean_FR_pyr_instantaneous = np.load(os.path.join(result_dir, 'mean_iFR_pyr_full.npy'))
    mean_FR_bc_instantaneous = np.load(os.path.join(result_dir, 'mean_iFR_bc_full.npy'))
    mean_FR_olm_instantaneous = np.load(os.path.join(result_dir, 'mean_iFR_olm_full.npy'))

    c_map_soma = matplotlib.colormaps['viridis']
    c_map_soma.set_bad('white')

    vmin = min(min(np.min(mean_FR_pyr_pop), np.min(mean_FR_bc_pop)), np.min(mean_FR_olm_pop))
    vmax = max(max(np.max(mean_FR_pyr_pop), np.max(mean_FR_bc_pop)), np.max(mean_FR_olm_pop))

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


    # make heatmaps
    X, Y = np.meshgrid(np.array([i for i in range(len(k_x))]), np.array([i for i in range(len(k_factors))]))

    pyr_pop = mean_FR_pyr_pop.copy()
    pyr_pop[pyr_pop==0.0] = np.nan

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(10,10))

    im_thr = ax.imshow(np.absolute(pyr_pop.transpose()), norm=norm, cmap=c_map_soma, origin='lower', aspect='auto', interpolation='bicubic')
    contours = plt.contour(X, Y, np.absolute(pyr_pop.transpose()), 8, colors='black', interpolation='bicubic')
    plt.clabel(contours, inline=1, fontsize=10)
    
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=c_map_soma),
    #             cax=ax, orientation='vertical', label='Firing rate (Hz)')
    cbar = fig.colorbar(im_thr, ax=ax)
    cbar.set_label("Firing rate (Hz)")
    ax.set_title('Mean firing rate of Pyramidal neurons over last second of simulation')
    ax.set_xticks([i for i in range(len(k_x))])
    ax.set_yticks([i for i in range(len(k_factors))])
    ax.set_xticklabels(k_x )
    ax.set_yticklabels(k_factors )
    ax.set_xlabel("E -> I")
    ax.set_ylabel("I -> E")
    print('saving figure...')
    plt.savefig(os.path.join(result_dir, 'mean_FR_pyr_full.png'), bbox_inches="tight")
    plt.close(fig)
    
    bc_pop = mean_FR_bc_pop.copy()
    bc_pop[bc_pop==0.0] = np.nan

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(10,10))

    im_thr = ax.imshow(np.absolute(bc_pop.transpose()), cmap=c_map_soma, norm=norm, origin='lower', aspect='auto', interpolation='bicubic')
    contours = plt.contour(X, Y, np.absolute(bc_pop.transpose()), 8, colors='black', interpolation='bicubic')
    plt.clabel(contours, inline=1, fontsize=10)
    cbar = fig.colorbar(im_thr, ax=ax)
    cbar.set_label("Firing rate (Hz)")
    ax.set_title('Mean firing rate of basket neurons over last second of simulation')
    ax.set_xticks([i for i in range(len(k_x))])
    ax.set_yticks([i for i in range(len(k_factors))])
    ax.set_xticklabels(k_x)
    ax.set_yticklabels(k_factors)
    ax.set_xlabel("E -> I")
    ax.set_ylabel("I -> E")
    print('saving figure...')
    plt.savefig(os.path.join(result_dir, 'mean_FR_bc_full.png'), bbox_inches="tight")
    plt.close(fig)


    olm_pop = mean_FR_olm_pop.copy()
    olm_pop[olm_pop==0.0] = np.nan

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(10,10))

    im_thr = ax.imshow(np.absolute(olm_pop.transpose()), cmap=c_map_soma, norm=norm, origin='lower', aspect='auto', interpolation='bicubic')
    contours = plt.contour(X, Y, np.absolute(olm_pop.transpose()), 8, colors='black', interpolation='bicubic')
    plt.clabel(contours, inline=1, fontsize=10)
    cbar = fig.colorbar(im_thr, ax=ax)
    cbar.set_label("Firing rate (Hz)")
    ax.set_title('Mean firing rate of OLM neurons over last second of simulation')
    ax.set_xticks([i for i in range(len(k_x))])
    ax.set_yticks([i for i in range(len(k_factors))])
    ax.set_xticklabels(k_x)
    ax.set_yticklabels(k_factors)
    ax.set_xlabel("E -> I")
    ax.set_ylabel("I -> E")
    print('saving figure...')
    plt.savefig(os.path.join(result_dir, 'mean_FR_olm_full.png'), bbox_inches="tight")
    plt.close(fig)

    sys.exit(0)
    #  instantaneous
    pyr_instant = mean_FR_pyr_instantaneous.copy()
    pyr_instant[pyr_instant==0.0] = np.nan

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(10,10))

    im_thr = ax.imshow(np.absolute(pyr_instant.transpose()), cmap=c_map_soma, vmin=vmin, vmax=vmax, origin='lower', aspect='auto', interpolation='bicubic')
    # contours = plt.contour(X, Y, np.absolute(pyr_instant.transpose()), 8, colors='azure', interpolation='bicubic')
    # plt.clabel(contours, inline=1, fontsize=10)
    cbar = fig.colorbar(im_thr, ax=ax)
    cbar.set_label("Firing rate (Hz)")
    ax.set_title('Mean instantaneous firing rate of pyramidal neurons over last second of simulation')
    ax.set_xticks([i for i in range(len(k_x))])
    ax.set_yticks([i for i in range(len(k_factors))])
    ax.set_xticklabels(k_x)
    ax.set_yticklabels(k_factors)
    ax.set_xlabel("E -> I")
    ax.set_ylabel("I -> E")
    print('saving figure...')
    plt.savefig(os.path.join(result_dir, 'mean_iFR_pyr_full.png'), bbox_inches="tight")
    plt.close(fig)


    bc_instant = mean_FR_bc_instantaneous.copy()
    bc_instant[bc_instant==0.0] = np.nan

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(10,10))

    im_thr = ax.imshow(np.absolute(bc_instant.transpose()), cmap=c_map_soma, vmin=vmin, vmax=vmax, origin='lower', aspect='auto', interpolation='bicubic')
    # contours = plt.contour(X, Y, np.absolute(bc_instant.transpose()), 8, colors='azure', interpolation='bicubic')
    # plt.clabel(contours, inline=1, fontsize=10)
    cbar = fig.colorbar(im_thr, ax=ax)
    cbar.set_label("Firing rate (Hz)")
    ax.set_title('Mean instantaneous firing rate of basket neurons over last second of simulation')
    ax.set_xticks([i for i in range(len(k_x))])
    ax.set_yticks([i for i in range(len(k_factors))])
    ax.set_xticklabels(k_x)
    ax.set_yticklabels(k_factors)
    ax.set_xlabel("E -> I")
    ax.set_ylabel("I -> E")
    print('saving figure...')
    plt.savefig(os.path.join(result_dir, 'mean_iFR_bc_full.png'), bbox_inches="tight")
    plt.close(fig)


    olm_instant = mean_FR_olm_instantaneous.copy()
    olm_instant[olm_instant==0.0] = np.nan

    fig, ax = plt.subplots(1, 1, sharey='row', figsize=(10,10))

    im_thr = ax.imshow(np.absolute(olm_instant.transpose()), cmap=c_map_soma, vmin=vmin, vmax=vmax, origin='lower', aspect='auto', interpolation='bicubic')
    # contours = plt.contour(X, Y, np.absolute(olm_instant.transpose()), 8, colors='azure', interpolation='bicubic')
    # plt.clabel(contours, inline=1, fontsize=10)
    cbar = fig.colorbar(im_thr, ax=ax)
    cbar.set_label("Firing rate (Hz)")
    ax.set_title('Mean instantaneous firing rate of OLM neurons over last second of simulation')
    ax.set_xticks([i for i in range(len(k_x))])
    ax.set_yticks([i for i in range(len(k_factors))])
    ax.set_xticklabels(k_x)
    ax.set_yticklabels(k_factors)
    ax.set_xlabel("E -> I")
    ax.set_ylabel("I -> E")
    print('saving figure...')
    plt.savefig(os.path.join(result_dir, 'mean_iFR_olm_full.png'), bbox_inches="tight")
    plt.close(fig)

