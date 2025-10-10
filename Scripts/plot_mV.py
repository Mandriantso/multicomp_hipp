import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def plot_Vm(time_vec, Vm_vec, cell_type, gid, Vm_last_vec=None, input=None, show_input=False):

    if show_input:
        if cell_type == "sca":
            fig, axs = plt.subplots(3, 1, figsize=(9, 3), sharex=True)
            # input
            axs[0].plot(input['t'], input['amplitude'][0], color='red')
            axs[0].set_ylabel("nA")
            axs[0].set_title('Input')
            # Vm first node
            axs[1].plot(time_vec, Vm_vec, color='grey')
            axs[1].set_ylabel("mV")
            axs[1].set_title('First node')
            # Vm last node
            axs[2].plot(time_vec, Vm_last_vec, color='black')
            axs[2].set_xlabel("Time (ms)")
            axs[2].set_ylabel("mV")
            axs[2].set_title('Last node')
            plt.setp([axs[1], axs[2]], ylim=(-90, 40))
        else:
            fig, axs = plt.subplots(2, 1, figsize=(9, 3))
            # input
            axs[0].plot(input['t'], input['amplitude'][0], color='red')
            axs[0].set_ylabel("nA")
            axs[0].set_title('Input')
            # Vm
            axs[1].plot(time_vec, Vm_vec, color='grey')
            axs[1].set_ylabel("mV")
            axs[1].set_title('First node')
            plt.setp([axs[1]], ylim=(-90, 40))
    else:
        if cell_type == "sca":
            fig, axs = plt.subplots(2, 1, figsize=(9, 3))
            # Vm first node
            axs[0].plot(time_vec, Vm_vec, color='grey')
            axs[0].set_ylabel("mV")
            axs[0].set_title('First node')
            # Vm last node
            axs[1].plot(time_vec, Vm_last_vec, color='black')
            axs[1].set_xlabel("Time (ms)")
            axs[1].set_ylabel("mV")
            axs[1].set_title('Last node')
            plt.setp([axs[1]], ylim=(-90, 40))
        else:
            fig, axs = plt.subplots(1, 1, figsize=(9, 3))
            # Vm
            axs.plot(time_vec, Vm_vec, color='grey')
            axs.set_ylabel("mV")
            axs.set_title('First node')
            plt.setp([axs], ylim=(-90, 40))

    plt.tight_layout()
    fig.suptitle(f'{cell_type}[{gid}]')


if __name__ == "__main__":
    ### set data directory name
    data_dir_name = "2025_10_10 13H46M57 no stim - v_init -80"
    script_dir = os.path.dirname(os.path.abspath(__file__)) # this dir
    parent_dir = Path(script_dir).parent

    results_dir = "network_with_schaffers_results"

    data_dir = os.path.join(parent_dir, results_dir, data_dir_name, "data")

    ### retrieve data
    cell_type = "sca" # pyr, bc, olm or sca
    show_input = True

    # input
    inputs = np.load(os.path.join(data_dir, "theta_inputs.npz"))

    # cell_type Vm
    cell_Vm = np.load(os.path.join(data_dir, f"CA1_{cell_type}_Vm.npz"))

    n_pyr_ca1 = 100
    n_bc_ca1 = 9
    n_olm_ca1 = 3
    n_schaffers_ca3 = 26

    n_cells_ca1 = n_pyr_ca1 + n_bc_ca1 + n_olm_ca1
    n_all_cells = n_cells_ca1 + n_schaffers_ca3

    # cell gids
    gids_pyr_soma = [2*n for n in range(n_pyr_ca1)]
    gids_pyr_axon = [2*n + 1 for n in range(n_pyr_ca1)]

    gids_interneurons = list(range(2*n_pyr_ca1, n_cells_ca1+n_pyr_ca1))

    gids_bc = [gid for gid in gids_interneurons if gid < 2*n_pyr_ca1 + n_bc_ca1]

    gids_olm = [gid for gid in gids_interneurons if gid >= 2*n_pyr_ca1 + n_bc_ca1]

    gids_sca_first_node = [n_cells_ca1+n_pyr_ca1+2*n for n in range(n_schaffers_ca3)]
    gids_sca_last_node = [n_cells_ca1+n_pyr_ca1+2*n + 1 for n in range(n_schaffers_ca3)]   

    # set gids
    if cell_type == "pyr":
        gids = gids_pyr_soma
    elif cell_type == "bc":
        gids = gids_bc
    elif cell_type == "olm":
        gids = gids_olm
    else:
        gids = gids_sca_first_node
        gids_last = gids_sca_last_node

    
    if cell_type == "sca":
        cell_last_Vm = np.load(os.path.join(data_dir, f"CA1_{cell_type}_last_Vm.npz"))


    for i in range(len(cell_Vm)-1):
        if show_input:
            if cell_type == "sca":
                plot_Vm(cell_Vm['time'], cell_Vm[str(gids[i])], cell_type, gids[i], cell_last_Vm[str(gids[i])], input=inputs, show_input=True)
            else:
                plot_Vm(cell_Vm['time'], cell_Vm[str(gids[i])], cell_type, gids[i], input=inputs, show_input=True)
        else:
            if cell_type == "sca":
                plot_Vm(cell_Vm['time'], cell_Vm[str(gids[i])], cell_type, gids[i], cell_last_Vm[str(gids[i])])
            else:
                plot_Vm(cell_Vm['time'], cell_Vm[str(gids[i])], cell_type, gids[i])
    plt.show()