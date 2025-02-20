import os
import sys
import time
from datetime import datetime
import argparse

from neuron import h
from Cells.Cells import PyramidalCell, BasketCell, OLMCell
# from Scripts.Stimulation import *
from Scripts.Network import *
from Scripts.myplot import save_raster, save_FR, save_specgram
from Scripts.utilities import *
from Scripts.anatomy import *
from Model import settings
import parameters

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
import random
from collections import OrderedDict


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


def _set_v_init():
    for cell_ in ca1_cells:
        r = random.uniform(-10, 10)
        for sec in cell_.all:
            for seg in sec:
                seg.v = settings.sim_v_init + r  


# Parse arguments
parser = argparse.ArgumentParser(description='Multicomp model parameters search')

parser.add_argument('-p', '--parameters',
                    nargs='?',
                    metavar='-p',
                    type=str,
                    default=os.path.join('configs', 'default_parameters_1.json'),
                    help='Parameters file (json format)')

parser.add_argument('-sd', '--save_dir',
                    nargs='?',
                    metavar='-sd',
                    type=str,
                    default='results_param_search',
                    help='Destination directory to save the results')

args = parser.parse_args()
filename = args.parameters
resdir = args.save_dir

try:
    data = parameters.load(filename)
    print('Using "{0}"'.format(filename))
except Exception as e:
    print(e)
    print('Using "default_parameters_1.json"')
    data = parameters._data
parameters.dump(data) # TODO: update file after changing weights ?
print()

# Settings initialization
settings.init(data)

RNG = np.random.default_rng()

# initialize MPI for parallel computing
h.nrnmpi_init()
pc = h.ParallelContext()
rank = pc.id()

# factor by which to multiply E -> I and I -> E weights
K_factors = [0.1, 0.25, 0.33, 0.5, 0.6, 0.75] 

if rank == 0:
    # Create directories
    print('\n[00] Making directories...')
    print('-'*32)
    sys.stdout.flush()

dirs = {}
dirs['results'] = resdir

if not os.path.isdir(dirs['results']) and rank == 0:
    print('[+] Creating directory', dirs['results'])
    sys.stdout.flush()
    os.makedirs(dirs['results'])

# Creating neuron populations
if rank == 0:
    print('\n[10] Making the neuron populations...')
    print('-'*32)

    print('[+] Loading coordinates')
    sys.stdout.flush()


# retrieve coordinates
coordinates_dir = 'Positions'
ca1_coordinates = os.path.join(coordinates_dir, 'ca1')

ca1_pyr_coordinates_flat = np.load(os.path.join(ca1_coordinates, 'pyr_coordinates_flat_constrained.npy'))
ca1_bc_coordinates_flat = np.load(os.path.join(ca1_coordinates, 'bc_coordinates_flat.npy'))
ca1_olm_coordinates_flat = np.load(os.path.join(ca1_coordinates, 'olm_coordinates_flat.npy'))

# sort by x
ca1_pyr_coordinates_flat = ca1_pyr_coordinates_flat[ca1_pyr_coordinates_flat[:,0].argsort()]
ca1_bc_coordinates_flat = ca1_bc_coordinates_flat[ca1_bc_coordinates_flat[:,0].argsort()]
ca1_olm_coordinates_flat = ca1_olm_coordinates_flat[ca1_olm_coordinates_flat[:,0].argsort()]

# create cells
n_pyr_ca1 = settings.N_CA1[0]
n_bc_ca1 = settings.N_CA1[1]
n_olm_ca1 = settings.N_CA1[2]

n_cells_ca1 = n_pyr_ca1 + n_bc_ca1 + n_olm_ca1 

# set gids
gids_pyr_soma = [2*n for n in range(pc.id(), n_pyr_ca1, pc.nhost())]
gids_pyr_axon = [2*n + 1 for n in range(pc.id(), n_pyr_ca1, pc.nhost())]

gids_interneurons = list(range(pc.id() + 2*n_pyr_ca1, n_cells_ca1+n_pyr_ca1, pc.nhost()))

gids_bc = [gid for gid in gids_interneurons if gid < 2*n_pyr_ca1 + n_bc_ca1]

gids_olm = [gid for gid in gids_interneurons if gid >= 2*n_pyr_ca1 + n_bc_ca1]

# associate gid to processor
for gid in gids_pyr_soma:
    pc.set_gid2node(gid, pc.id())

for gid in gids_pyr_axon:
    pc.set_gid2node(gid, pc.id())

for gid in gids_interneurons:
    pc.set_gid2node(gid, pc.id())

if rank == 0:  
    print('[+] Creating cells')
    sys.stdout.flush()

ca1_pyr_cells = []
for gid_soma, gid_axon in zip(gids_pyr_soma, gids_pyr_axon):
    cell_ = PyramidalCell(gid_soma=gid_soma, gid_axon=gid_axon, x=ca1_pyr_coordinates_flat[int(gid_soma/2), 0], y=ca1_pyr_coordinates_flat[int(gid_soma/2), 1])
    ca1_pyr_cells.append(cell_)
    # associate gid to spike_detector
    pc.cell(gid_soma, cell_._spike_detector)
    pc.cell(gid_axon, cell_._spike_detector_axon)

ca1_bc_cells = []
for gid in gids_bc:
    cell_ = BasketCell(gid=gid, x=ca1_bc_coordinates_flat[gid - 2*n_pyr_ca1,0], y=ca1_bc_coordinates_flat[gid - 2*n_pyr_ca1,1])
    ca1_bc_cells.append(cell_)
    # associate gid to spike_detector
    pc.cell(gid, cell_._spike_detector)

ca1_olm_cells = []
for gid in gids_olm:
    cell_ = OLMCell(gid=gid, x=ca1_olm_coordinates_flat[gid - 2*n_pyr_ca1 - n_bc_ca1,0], y=ca1_olm_coordinates_flat[gid - 2*n_pyr_ca1 - n_bc_ca1,1])
    ca1_olm_cells.append(cell_)
    # associate gid to spike_detector
    pc.cell(gid, cell_._spike_detector)

ca1_cells = ca1_pyr_cells + ca1_bc_cells + ca1_olm_cells

# adding noise
if rank == 0:  
    print('[+] Adding noise current...')
    sys.stdout.flush()

i_noise = h.Vector()

for cell_ in ca1_cells:
    for sec in cell_.all:
        sec.insert("Inoise")
        sec.myseed_Inoise = RNG.integers(0, 1000000000000000, 1) #seed_noise #settings.seed_noise
        if "PyramidalCell" in str(cell_):
            sec.sigma_Inoise = settings.sigma_CA1[0]
            sec.mean_Inoise = settings.mean_CA1[0]
            sec.tau_Inoise = settings.tau_CA1[0]
            sec.inoise_Inoise = -0.0
            i_noise.record(cell_.soma(0.5)._ref_inoise_Inoise)
        elif "OLMCell" in str(cell_):
            sec.sigma_Inoise = settings.sigma_CA1[2]
            sec.mean_Inoise = settings.mean_CA1[2]
            sec.tau_Inoise = settings.tau_CA1[2]
            sec.inoise_Inoise = -0.0
        else:
            sec.sigma_Inoise = settings.sigma_CA1[1]
            sec.mean_Inoise = settings.mean_CA1[1]
            sec.tau_Inoise = settings.tau_CA1[1]
            sec.inoise_Inoise = -0.0

    # remove noise in axon
    if hasattr(cell_, "axonal"):
        mt = h.MechanismType(0)
        mt.select('Inoise')
        for sec in cell_.axonal:
            mt.remove(sec=sec)

    
# set connections
if rank == 0:  
    print('\n[11] Setting connections...')
    print('-'*32)

    print('[+] Finding all postsynaptic and presynaptic cells...')
    sys.stdout.flush()

# set connection matrix
conn_mat = np.zeros((n_cells_ca1, n_cells_ca1))

for i in range(n_pyr_ca1):
    # for j in range(n_pyr_ca1):
    #     dist_value = np.sqrt((ca1_pyr_coordinates_flat[i, 0] - ca1_pyr_coordinates_flat[j, 0])**2 + (ca1_pyr_coordinates_flat[i, 1] - ca1_pyr_coordinates_flat[j, 1])**2)
    #     if settings.syn_dist_CA1[0] >= dist_value and i != j and conn_mat[i, :n_pyr_ca1].sum(axis=0) < 1 and conn_mat[:n_pyr_ca1, j].sum(axis=0) < 1:
    #         conn_mat[i, j] = 1

    for j in range(n_bc_ca1):
        dist_value = np.sqrt((ca1_pyr_coordinates_flat[i, 0] - ca1_bc_coordinates_flat[j, 0])**2 + (ca1_pyr_coordinates_flat[i, 1] - ca1_bc_coordinates_flat[j, 1])**2)
        if settings.syn_dist_CA1[0] >= dist_value:
            conn_mat[i, j + n_pyr_ca1] = 1

    for j in range(n_olm_ca1):
        dist_value = np.sqrt((ca1_pyr_coordinates_flat[i, 0] - ca1_olm_coordinates_flat[j, 0])**2 + (ca1_pyr_coordinates_flat[i, 1] - ca1_olm_coordinates_flat[j, 1])**2)
        if settings.syn_dist_CA1[0] >= dist_value:
            conn_mat[i, j + n_pyr_ca1 + n_bc_ca1] = 1

for i in range(n_bc_ca1):
    for j in range(n_pyr_ca1):
        dist_value = np.sqrt((ca1_bc_coordinates_flat[i, 0] - ca1_pyr_coordinates_flat[j, 0])**2 + (ca1_bc_coordinates_flat[i, 1] - ca1_pyr_coordinates_flat[j, 1])**2)
        if settings.syn_dist_CA1[1] >= dist_value and conn_mat[n_pyr_ca1:n_pyr_ca1+n_bc_ca1, j].sum(axis=0) < 1: # only one conn. from BC to Pyr
            conn_mat[i + n_pyr_ca1, j] = 1

    for j in range(n_bc_ca1):
        dist_value = np.sqrt((ca1_bc_coordinates_flat[i, 0] - ca1_bc_coordinates_flat[j, 0])**2 + (ca1_bc_coordinates_flat[i, 1] - ca1_bc_coordinates_flat[j, 1])**2)
        if settings.syn_dist_CA1[1] >= dist_value and i != j:
            conn_mat[i + n_pyr_ca1, j + n_pyr_ca1] = 1

for i in range(n_olm_ca1):
    for j in range(n_pyr_ca1):
        dist_value = np.sqrt((ca1_olm_coordinates_flat[i, 0] - ca1_pyr_coordinates_flat[j, 0])**2 + (ca1_olm_coordinates_flat[i, 1] - ca1_pyr_coordinates_flat[j, 1])**2)
        if settings.syn_dist_CA1[2] >= dist_value:
            conn_mat[i + n_pyr_ca1 + n_bc_ca1, j] = 1

# fill postsynaptic and presynaptic neurons
for i in range(n_pyr_ca1):
    for j in range(n_pyr_ca1):
        if conn_mat[i, j] > 0 and pc.gid_exists(2*i):
            pc.gid2cell(2*i)._postsyn_list.append(2*j)
        if conn_mat[i, j] > 0 and pc.gid_exists(2*j):
            pc.gid2cell(2*j)._presyn_list.append(2*i)
    
    for j in range(n_pyr_ca1, n_pyr_ca1 + n_bc_ca1):
        if conn_mat[i, j] > 0 and pc.gid_exists(2*i):
            pc.gid2cell(2*i)._postsyn_list.append(n_pyr_ca1 + j)
        if conn_mat[i, j] > 0 and pc.gid_exists(n_pyr_ca1 + j):
            pc.gid2cell(n_pyr_ca1 + j)._presyn_list.append(2*i)

    for j in range(n_pyr_ca1 + n_bc_ca1, n_cells_ca1):
        if conn_mat[i, j] > 0 and pc.gid_exists(2*i):
            pc.gid2cell(2*i)._postsyn_list.append(n_pyr_ca1 + j)
        if conn_mat[i, j] > 0 and pc.gid_exists(n_pyr_ca1 + j):
            pc.gid2cell(n_pyr_ca1 + j)._presyn_list.append(2*i)


for i in range(n_pyr_ca1, n_pyr_ca1 + n_bc_ca1):
    for j in range(n_pyr_ca1):
        if conn_mat[i, j] > 0 and pc.gid_exists(n_pyr_ca1 + i):
            pc.gid2cell(n_pyr_ca1 + i)._postsyn_list.append(2*j)
        if conn_mat[i, j] > 0 and pc.gid_exists(2*j):
            pc.gid2cell(2*j)._presyn_list.append(n_pyr_ca1 + i)
    
    for j in range(n_pyr_ca1, n_pyr_ca1 + n_bc_ca1):
        if conn_mat[i, j] > 0 and pc.gid_exists(n_pyr_ca1 + i):
            pc.gid2cell(n_pyr_ca1 + i)._postsyn_list.append(n_pyr_ca1 + j)
        if conn_mat[i, j] > 0 and pc.gid_exists(n_pyr_ca1 + j):
            pc.gid2cell(n_pyr_ca1 + j)._presyn_list.append(n_pyr_ca1 + i)
    
for i in range(n_pyr_ca1 + n_bc_ca1, n_cells_ca1):
    for j in range(n_pyr_ca1):
        if conn_mat[i, j] > 0 and pc.gid_exists(n_pyr_ca1 + i):
            pc.gid2cell(n_pyr_ca1 + i)._postsyn_list.append(2*j)
        if conn_mat[i, j] > 0 and pc.gid_exists(2*j):
            pc.gid2cell(2*j)._presyn_list.append(n_pyr_ca1 + i)

if rank == 0:
    print('[+] Found all postsynatpic cells')
    sys.stdout.flush()

# wait for all processors to reach this point
pc.barrier()

if rank == 0:
    print('\n[01] Starting iterations...')
    print('-'*32)
    sys.stdout.flush()

for k_e in K_factors: # for Pyr -> BC weights
    if rank == 0:
        print(f'\n[+]    k_e = {k_e}')
        print('-'*32)
        sys.stdout.flush()

    dirs['dir_E'] = os.path.join(dirs['results'], "w_E_{}".format(k_e))
    if not os.path.isdir(dirs['dir_E']) and rank == 0:
        print('[+] Creating directory', dirs['dir_E'])
        sys.stdout.flush()
        os.makedirs(dirs['dir_E'])

    for k_i in K_factors: # for BC -> Pyr weights
        if rank == 0:
            print(f'\n[+]    k_i = {k_i}')
            print('-'*32)
            sys.stdout.flush()

        dirs['save_dir'] = os.path.join(dirs['dir_E'], "w_I_{}".format(k_i))
        if not os.path.isdir(dirs['save_dir']) and rank == 0:
            print('[+] Creating directory', dirs['save_dir'])
            sys.stdout.flush()
            os.makedirs(dirs['save_dir'])


        dirs['data'] = os.path.join(dirs['save_dir'], 'data')
        if not os.path.isdir(dirs['data']) and rank == 0:
            print('[+] Creating directory', dirs['data'])
            sys.stdout.flush()
            os.makedirs(dirs['data'])

        dirs['coords'] = os.path.join(dirs['data'], 'coordinates')
        if not os.path.isdir(dirs['coords']) and rank == 0:
            print('[+] Creating directory', dirs['coords'])
            sys.stdout.flush()
            os.makedirs(dirs['coords'])

        dirs['figures'] = os.path.join(dirs['save_dir'], 'figures')
        if not os.path.isdir(dirs['figures']) and rank == 0:
            print('[+] Creating directory', dirs['figures'])
            sys.stdout.flush()
            os.makedirs(dirs['figures'])

        if rank == 0:
            print('[+] Saving connection matrix')
            sys.stdout.flush()
            np.save(os.path.join(dirs['data'], "connection_matrix.npy"), conn_mat)

        #starting from here
        if rank == 0:
            print('[+] Set connection weight...')
            sys.stdout.flush()

        # set connections to Pyramidal cells
        for cell_ in ca1_pyr_cells: # to | from
            for pregid in cell_._presyn_list:
                if pregid < 2*n_pyr_ca1: # from Pyramidal
                    target_secs = list(cell_.proximal)
                    target_sec = random.choice(target_secs)
                    mt_ = h.MechanismType(1)
                    mt_.select("Exp2Syn")
                    pp = mt_.pp_begin(sec=target_sec)
                    nc_ = pc.gid_connect(pregid, pp)
                    nc_.weight[0] = settings.w_CA1[0][0]
                    nc_.threshold = settings.syn_threshold
                    nc_.delay = settings.syn_delay
                    cell_._ncs.append(nc_)
                elif pregid >= 2*n_pyr_ca1 and pregid < 2*n_pyr_ca1 + n_bc_ca1: # from Basket
                    target_secs = list(cell_.somatic)
                    target_sec = random.choice(target_secs)
                    mt_ = h.MechanismType(1)
                    mt_.select("Exp2Syn")
                    pp = mt_.pp_begin(sec=target_sec)
                    nc_ = pc.gid_connect(pregid, pp)
                    nc_.weight[0] = settings.w_CA1[1][0] * k_i
                    nc_.threshold = settings.syn_threshold
                    nc_.delay = settings.syn_delay
                    cell_._ncs.append(nc_)
                else: # from OLM
                    target_secs = list(cell_.lm_list)
                    target_sec = random.choice(target_secs)
                    mt_ = h.MechanismType(1)
                    mt_.select("Exp2Syn")
                    pp = mt_.pp_begin(sec=target_sec)
                    nc_ = pc.gid_connect(pregid, pp)
                    nc_.weight[0] = settings.w_CA1[2][0]
                    nc_.threshold = settings.syn_threshold
                    nc_.delay = settings.syn_delay
                    cell_._ncs.append(nc_)

        # set connections to Basket cells
        for cell_ in ca1_bc_cells:
            for pregid in cell_._presyn_list:
                if pregid < 2*n_pyr_ca1: # from Pyramidal
                    target_secs = list(cell_.proximal_apical)
                    target_sec = random.choice(target_secs)
                    mt_ = h.MechanismType(1)
                    mt_.select("Exp2Syn")
                    pp = mt_.pp_begin(sec=target_sec)
                    nc_ = pc.gid_connect(pregid, pp)
                    nc_.weight[0] = settings.w_CA1[0][1] * k_e
                    nc_.threshold = settings.syn_threshold
                    nc_.delay = settings.syn_delay
                    cell_._ncs.append(nc_)
                elif pregid >= 2*n_pyr_ca1 and pregid < 2*n_pyr_ca1 + n_bc_ca1: # from Basket
                    target_secs = list(cell_.somatic)
                    target_sec = random.choice(target_secs)
                    mt_ = h.MechanismType(1)
                    mt_.select("Exp2Syn")
                    pp = mt_.pp_begin(sec=target_sec)
                    nc_ = pc.gid_connect(pregid, pp)
                    nc_.weight[0] = settings.w_CA1[1][1]
                    nc_.threshold = settings.syn_threshold
                    nc_.delay = settings.syn_delay
                    cell_._ncs.append(nc_)

        # set connections to OLM cells
        for cell_ in ca1_olm_cells:
            for pregid in cell_._presyn_list:
                if pregid < 2*n_pyr_ca1: # from Pyramidal
                    target_secs = list(cell_.basal)
                    target_sec = random.choice(target_secs)
                    mt_ = h.MechanismType(1)
                    mt_.select("Exp2Syn")
                    pp = mt_.pp_begin(sec=target_sec)
                    nc_ = pc.gid_connect(pregid, pp)
                    nc_.weight[0] = settings.w_CA1[0][2]
                    nc_.threshold = settings.syn_threshold
                    nc_.delay = settings.syn_delay
                    cell_._ncs.append(nc_)

        # create inputs
        # inputs are set as intracellular ramp current to soma of pyramidal cells
        if rank == 0:
            print('[+] Connecting cells done')
            print('\n[20] Setting the inputs...')
            sys.stdout.flush()

        # setting current vectors
        VecT = h.Vector([0, settings.duration])
        VecStim = h.Vector([0., 1.])
        stim_amp = h.Vector()

        for cell_ in ca1_cells:
            if 'PyramidalCell' in str(cell_):
                stim_ = h.IClamp(cell_.soma(0.5))
                stim_.delay = 0
                stim_.dur = 1e9

                VecStim.play(stim_._ref_amp, VecT, 1)
                cell_._inputs_list.append(stim_)

                stim_amp.record(stim_._ref_i) # last stim_

        if rank == 0:
            print('[+] Inputs done')
            print('\n[30] Simulation...')
            print('-'*32)

            print('[+] Setting recording vectors')
            sys.stdout.flush()

        # Set recording vectors
        t_vec = h.Vector().record(h._ref_t)

        # set finitializehandler
        fih = h.FInitializeHandler(1, _set_v_init) # random initialization of Vm

        h.tstop = settings.duration
        h.celsius = 35

        pc.set_maxstep(10)

        if rank == 0:
            print('[+] Running simulation...')
            sys.stdout.flush()

        h.stdinit()
        h.cvode_active(0)
        start_time = time.time()
        pc.psolve(settings.duration)
        end_time = time.time()

        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)

        if rank == 0:
            print("Elapsed time : {:0>2} h {:0>2} min {:05.2f} s".format(int(hours),int(minutes),seconds))
            
            print('\n[31] Saving results...')
            print('-'*32)
            print('[+] Saving membrane potential vectors...')
            sys.stdout.flush()

        # retrieve results on local processor
        # local membrane potential
        local_potential_pyr = {cell_._gid: np.array(cell_.soma_v) for cell_ in ca1_pyr_cells}
        local_potential_bc = {cell_._gid: np.array(cell_.soma_v) for cell_ in ca1_bc_cells}
        local_potential_olm = {cell_._gid: np.array(cell_.soma_v) for cell_ in ca1_olm_cells}

        # local intracellular_current
        local_current_pyr = {cell_._gid: np.array(cell_.soma_i) for cell_ in ca1_pyr_cells}
        local_current_bc = {cell_._gid: np.array(cell_.soma_i) for cell_ in ca1_bc_cells}
        local_current_olm = {cell_._gid: np.array(cell_.soma_i) for cell_ in ca1_olm_cells}

        # local spike times
        local_spikes_pyr = {cell_._gid: list(cell_.spike_times) for cell_ in ca1_pyr_cells}
        local_spikes_bc = {cell_._gid: list(cell_.spike_times) for cell_ in ca1_bc_cells}
        local_spikes_olm = {cell_._gid: list(cell_.spike_times) for cell_ in ca1_olm_cells}

        # send results to all processors
        all_potential_pyr = pc.py_alltoall([local_potential_pyr] + [None] * (pc.nhost() - 1))
        all_potential_bc = pc.py_alltoall([local_potential_bc] + [None] * (pc.nhost() - 1))
        all_potential_olm = pc.py_alltoall([local_potential_olm] + [None] * (pc.nhost() - 1))

        all_current_pyr = pc.py_alltoall([local_current_pyr] + [None] * (pc.nhost() - 1))
        all_current_bc = pc.py_alltoall([local_current_bc] + [None] * (pc.nhost() - 1))
        all_current_olm = pc.py_alltoall([local_current_olm] + [None] * (pc.nhost() - 1))

        all_spikes_pyr = pc.py_alltoall([local_spikes_pyr] + [None] * (pc.nhost() - 1))
        all_spikes_bc = pc.py_alltoall([local_spikes_bc] + [None] * (pc.nhost() - 1))
        all_spikes_olm = pc.py_alltoall([local_spikes_olm] + [None] * (pc.nhost() - 1))

        if rank == 0:
            # combine the data from the various processors
            potential_pyr = {}
            potential_bc = {}
            potential_olm = {}

            current_pyr = {}
            current_bc = {}
            current_olm = {}

            spikes_pyr = {}
            spikes_bc = {}
            spikes_olm = {}

            for data in all_potential_pyr:
                potential_pyr.update(data)

            for data in all_potential_bc:
                potential_bc.update(data)

            for data in all_potential_olm:
                potential_olm.update(data)

            for data in all_current_pyr:
                current_pyr.update(data)

            for data in all_current_bc:
                current_bc.update(data)

            for data in all_current_olm:
                current_olm.update(data)

            for data in all_spikes_pyr:
                spikes_pyr.update(data)
            
            for data in all_spikes_bc:
                spikes_bc.update(data)

            for data in all_spikes_olm:
                spikes_olm.update(data)

            # arrange spikes
            id_spikes_pyr = []
            t_spikes_pyr = []
            for key, value in spikes_pyr.items():
                id_spikes_pyr.extend([key/2] * len(value))
                t_spikes_pyr.extend(value)

            id_spikes_bc = []
            t_spikes_bc = []
            for key, value in spikes_bc.items():
                id_spikes_bc.extend([key-n_pyr_ca1] * len(value))
                t_spikes_bc.extend(value)

            id_spikes_olm = []
            t_spikes_olm = []
            for key, value in spikes_olm.items():
                id_spikes_olm.extend([key-n_pyr_ca1] * len(value))
                t_spikes_olm.extend(value)

            # spectrogram
            cmesh_list = []
            vlow = []
            vhigh = []

            winsize_fr = 5 #ms
            overlap_fr = 0.9

            window_size = 1 #ms

            window_size_Pxx = 1000
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
            t_FR_pyr, count_pyr, FR_pyr, fs_n = compute_FR(np.array(t_spikes_pyr)*1e-3, n_pyr_ca1, settings.duration*1e-3, winsize_fr*1e-3, overlap_fr)
            t_FR_bc, count_bc, FR_bc, _ = compute_FR(np.array(t_spikes_bc)*1e-3, n_bc_ca1, settings.duration*1e-3, winsize_fr*1e-3, overlap_fr)
            t_FR_olm, count_olm, FR_olm, _ = compute_FR(np.array(t_spikes_olm)*1e-3, n_olm_ca1, settings.duration*1e-3, winsize_fr*1e-3, overlap_fr)

            with open(os.path.join(dirs["save_dir"], "logs.txt"), "w") as f:
                f.write("Simulation parameters\n")
                f.write("-------------------------\n")
                f.write("remark :\n")
                f.write("- no Pyr - Pyr connections\n")
                f.write("- BC - Pyr constrained to one connection max\n")
                f.write("\nPyr - BC weight : {}\n".format(settings.w_CA1[0][1] * k_e))
                f.write("BC - Pyr weight : {}\n".format(settings.w_CA1[1][0] * k_i))
                f.write("BC - BC weight : {}\n".format(settings.w_CA1[1][1]))
                f.write("OLM - Pyr weight : {}\n".format(settings.w_CA1[2][0]))
                f.write("Pyr - OLM weight : {}\n".format(settings.w_CA1[0][2]))

                f.write("\n\nSimulation results\n")
                f.write("-------------------------\n")
                f.write("Mean firing rate over simulation\n")
                f.write("Pyr firing rate : {} Hz\n".format(np.mean(FR_pyr)))
                f.write("BC firing rate : {} Hz\n".format(np.mean(FR_bc)))
                f.write("OLM firing rate : {} Hz\n\n".format(np.mean(FR_olm)))
                f.write("Mean firing rate over last 1s\n")
                f.write("Pyr firing rate : {} Hz\n".format(np.mean(FR_pyr[-200:])))
                f.write("BC firing rate : {} Hz\n".format(np.mean(FR_bc[-200:])))
                f.write("OLM firing rate : {} Hz\n\n".format(np.mean(FR_olm[-200:])))
                f.close()

            # compute spectrograms
            fv_pyr, tv_pyr, pspec_pyr = my_specgram2(FR_pyr, fs_n, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
            fv_bc, tv_bc, pspec_bc = my_specgram2(FR_bc, fs_n, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
            fv_olm, tv_olm, pspec_olm = my_specgram2(FR_olm, fs_n, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)

            vlow.append(pspec_pyr.min())
            vhigh.append(pspec_pyr.max())

            for cmsh in make_flat(cmesh_list):
                cmsh.set_clim(min(vlow), max(vhigh))

            save_specgram(os.path.join(dirs['figures'], 'specgram.png'), [tv_pyr, tv_bc, tv_olm], [fv_pyr, fv_bc, fv_olm], [pspec_pyr, pspec_bc, pspec_olm], ["pyramidal cells", "basket cells", "olm cells"])
            save_specgram(os.path.join(dirs['figures'], 'specgram_0_50.png'), [tv_pyr, tv_bc, tv_olm], [fv_pyr, fv_bc, fv_olm], [pspec_pyr, pspec_bc, pspec_olm], ["pyramidal cells", "basket cells", "olm cells"], ylim=[0, 50])

            # save vectors
            save_membrane_potential(os.path.join(dirs['data'], 'CA1_pyr_Vm.npz'), np.array(t_vec), potential_pyr) 
            save_membrane_potential(os.path.join(dirs['data'], 'CA1_bc_Vm.npz'), np.array(t_vec), potential_bc) 
            save_membrane_potential(os.path.join(dirs['data'], 'CA1_olm_Vm.npz'), np.array(t_vec), potential_olm) 

            save_membrane_potential(os.path.join(dirs['data'], 'CA1_pyr_i.npz'), np.array(t_vec), current_pyr) 
            save_membrane_potential(os.path.join(dirs['data'], 'CA1_bc_i.npz'), np.array(t_vec), current_bc) 
            save_membrane_potential(os.path.join(dirs['data'], 'CA1_olm_i.npz'), np.array(t_vec), current_olm) 

            np.savez(os.path.join(dirs['data'], 'CA1_pyr_spikemon.npz'), cell_id=np.array(id_spikes_pyr), t_spike=np.array(t_spikes_pyr))
            np.savez(os.path.join(dirs['data'], 'CA1_bc_spikemon.npz'), cell_id=np.array(id_spikes_bc), t_spike=np.array(t_spikes_bc))
            np.savez(os.path.join(dirs['data'], 'CA1_olm_spikemon.npz'), cell_id=np.array(id_spikes_olm), t_spike=np.array(t_spikes_olm))

            save_raster(os.path.join(dirs['figures'], 'raster_plot.png'), [t_spikes_pyr, t_spikes_bc, t_spikes_olm],
                        [id_spikes_pyr, id_spikes_bc, id_spikes_olm], 
                        ['skyblue', 'lightpink', 'darkorange'], ['pyramidal cells', 'basket cells', 'olm cells'],
                        x_lim=[0, settings.duration])
        
            save_raster(os.path.join(dirs['figures'], 'raster_plot_4750_5000.png'), [t_spikes_pyr, t_spikes_bc, t_spikes_olm],
                        [id_spikes_pyr, id_spikes_bc, id_spikes_olm], 
                        ['skyblue', 'lightpink', 'darkorange'], ['pyramidal cells', 'basket cells', 'olm cells'],
                        x_lim=[settings.duration - 1e3, settings.duration], size_raster=1.) # last second

            np.savez(os.path.join(dirs['data'], 'ramping_current.npz'), t=np.array(t_vec), amplitude=np.array(stim_amp))
            np.savez(os.path.join(dirs['data'], 'i_noise.npz'), t=np.array(t_vec), noise=np.array(i_noise))

            fig2, ax2 = plt.subplots(1,1,figsize=(9,3))
            ax2.plot(np.array(t_vec), np.array(stim_amp), color='red')
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("nA")
            plt.savefig(os.path.join(dirs['figures'], 'ramping_current.png'), bbox_inches="tight")

            fig3, ax3 = plt.subplots(1,1,figsize=(9,3))
            ax3.plot(np.array(t_vec), np.array(i_noise), color='black')
            ax3.set_xlabel("Time (ms)")
            ax3.set_ylabel("nA")
            plt.savefig(os.path.join(dirs['figures'], 'i_noise.png'), bbox_inches="tight")

            print("[+] Saving data done !")
            print("[+] Resizing vectors")
            sys.stdout.flush()

        # wait for all processors to reach this stage
        pc.barrier()

        # format vectors
        t_vec.resize(0)
        stim_amp.resize(0)
        i_noise.resize(0)

        for cell_ in ca1_cells:
            cell_.soma_v.resize(0)
            cell_.spike_times.resize(0)
            cell_.soma_i.resize(0)
            # cell_._spike_detector.resize(0)
            cell_._ncs = []
            cell_._inputs_list = []
    

        pc.barrier()
        # pc.gid_clear()

        if rank == 0:
            print('Next iteration !')
            sys.stdout.flush()


pc.done()
h.quit()
sys.exit(0)


