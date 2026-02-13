import os
import sys
import time
import argparse

from neuron import h
from Cells.Cells import PyramidalCell, BasketCell, OLMCell, SchafferCollateral_2
from Scripts.Stimulation import *
from Scripts.Network import *
from Scripts.utilities import *
from Scripts.anatomy import *
from Scripts.input import *
from Model import settings
import parameters

import numpy as np
import random


def _set_v_init():
    for cell_ in all_cells:
        r = random.uniform(-10, 10)
        if "SchafferCollateral" in str(cell_):
            for sec in cell_.all:
                for seg in sec:
                    seg.v = -80 + r #settings.sim_v_init + r#-80 #+ r  #TODO: specific vrest for each cell type to add in parameters ?
        else:
            for sec in cell_.all:
                for seg in sec:
                    seg.v = settings.sim_v_init + r


# Parse arguments
parser = argparse.ArgumentParser(description='Multicomp model with schaffer collaterals. Test amp effects with different modalities of stimulation and mainly biphasic train pulses at 50Hz')

parser.add_argument('-p', '--parameters',
                    nargs='?',
                    metavar='PARAM_FILE',
                    type=str,
                    default=os.path.join('configs', 'fixed_parameters_monopolar_stim_inner_CA1_biphasic_train_pulses.json'), 
                    help='Parameters file (json format)')

parser.add_argument('-sd', '--save_dir',
                    nargs='?',
                    metavar='SAVING_DIR',
                    type=str,
                    default='results_amp_variation',
                    help='Destination directory to save the results')

parser.add_argument('-a', '--amp',
                    nargs='?',
                    metavar='AMP',
                    type=float,
                    default=None,
                    help='Amplitude value to override in parameters')

args = parser.parse_args()
filename = args.parameters
resdir = args.save_dir

try:
    data = parameters.load(filename)
    print('Using "{0}"'.format(filename))
except Exception as e:
    print(e)
    print('Using "fixed_parameters_monopolar_stim_inner_CA1_biphasic_train_pulses.json"')
    data = parameters._data
parameters.dump(data) 
print()

# Settings initialization
settings.init(data)

# Check if amp is specified
if args.amp is not None:
    settings.stim_amp = args.amp


# Extract stimulation typ
stim_params = [settings.stim_electrode, settings.stim_type, settings.stim_waveform]
if settings.stim_electrode == "bipolar":
    stim_orientation = os.path.basename(args.parameters).split('_')[4:6]
    stim_orientation = '_'.join(stim_orientation)
else:
    stim_orientation = os.path.basename(args.parameters).split('_')[4]
stim_params.append(stim_orientation)
stim_type = '_'.join(stim_params)

print("json file : ", filename)
print("save_dir : ", resdir)
print("amp : ", settings.stim_amp)
print("stim_type : ", stim_type)

# for watermaks on figures -> reproducibility
git_kwargs = {'timestamp': time.ctime(),
              'branch': get_git_revision_branch(), 
              'hash': get_git_revision_hash(),
              'script_name': os.path.basename(__file__),
              'config_file': filename}

RNG = np.random.default_rng()

# initialize MPI for parallel computing
h.nrnmpi_init()
pc = h.ParallelContext()
rank = pc.id()

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


dirs['stim_dir'] = os.path.join(dirs['results'], stim_type)
if not os.path.isdir(dirs['stim_dir']) and rank == 0:
    print('[+] Creating directory', dirs['stim_dir'])
    sys.stdout.flush()
    os.makedirs(dirs['stim_dir'])


dirs['save_dir'] = os.path.join(dirs['stim_dir'], f"{settings.stim_amp}_mA")
if not os.path.isdir(dirs['save_dir']) and rank == 0:
    print('[+] Creating directory', dirs['save_dir'])
    sys.stdout.flush()
    os.makedirs(dirs['save_dir'])


dirs['data'] = os.path.join(dirs['save_dir'], 'data')
if not os.path.isdir(dirs['data']) and rank == 0:
    print('[+] Creating directory', dirs['data'])
    sys.stdout.flush()
    os.makedirs(dirs['data'])


dirs['figures'] = os.path.join(dirs['save_dir'], 'figures')
if not os.path.isdir(dirs['figures']) and rank == 0:
    print('[+] Creating directory', dirs['figures'])
    sys.stdout.flush()
    os.makedirs(dirs['figures'])


if rank == 0:
    print('\n[10] Making the neuron populations...')
    print('-'*32)

    print('[+] Loading coordinates')
    sys.stdout.flush()


# retrieve coordinates
coordinates_dir = 'positions_correct_layers_thickness'
ca1_coordinates = os.path.join(coordinates_dir, 'ca1')
ca3_coordinates = os.path.join(coordinates_dir, 'ca3')

ca1_pyr_coords = np.load(os.path.join(ca1_coordinates, 'pyr_coordinates.npy'))
ca1_bc_coords = np.load(os.path.join(ca1_coordinates, 'bc_coordinates.npy'))
ca1_olm_coords = np.load(os.path.join(ca1_coordinates, 'olm_coordinates.npy'))

ca3_schaffer_nodes_coordinates = np.load(os.path.join(ca3_coordinates, 'nodes_coordinates_equidistant.npy'), allow_pickle=True)
ca3_schaffer_xs_intrinsic = np.load(os.path.join(ca3_coordinates, 'xs_intrinsic_last_node_equidistant.npy'), allow_pickle=True)
ca3_schaffer_ys_intrinsic = np.load(os.path.join(ca3_coordinates, 'ys_intrinsic_last_node_equidistant.npy'), allow_pickle=True)
ca3_schaffer_xs_flat = np.load(os.path.join(ca3_coordinates, 'xs_flat_last_node_equidistant.npy'), allow_pickle=True)
ca3_schaffer_ys_flat = np.load(os.path.join(ca3_coordinates, 'ys_flat_last_node_equidistant.npy'), allow_pickle=True)

# create cells
n_pyr_ca1 = settings.N_CA1[0]
n_bc_ca1 = settings.N_CA1[1]
n_olm_ca1 = settings.N_CA1[2]

n_cells_ca1 = n_pyr_ca1 + n_bc_ca1 + n_olm_ca1 

n_schaffers_ca3 = settings.N_CA3[0]

n_all_cells = n_cells_ca1 + n_schaffers_ca3

# set gids
gids_pyr_soma = [2*n for n in range(pc.id(), n_pyr_ca1, pc.nhost())]
gids_pyr_axon = [2*n + 1 for n in range(pc.id(), n_pyr_ca1, pc.nhost())]

gids_interneurons = list(range(pc.id() + 2*n_pyr_ca1, n_cells_ca1+n_pyr_ca1, pc.nhost()))

gids_bc = [gid for gid in gids_interneurons if gid < 2*n_pyr_ca1 + n_bc_ca1]

gids_olm = [gid for gid in gids_interneurons if gid >= 2*n_pyr_ca1 + n_bc_ca1]

gids_sca_first_node = [n_cells_ca1+n_pyr_ca1+2*n for n in range(pc.id(), n_schaffers_ca3, pc.nhost())]
gids_sca_last_node = [n_cells_ca1+n_pyr_ca1+2*n + 1 for n in range(pc.id(), n_schaffers_ca3, pc.nhost())]

# associate gid to processor
for gid in gids_pyr_soma:
    pc.set_gid2node(gid, pc.id())

for gid in gids_pyr_axon:
    pc.set_gid2node(gid, pc.id())

for gid in gids_interneurons:
    pc.set_gid2node(gid, pc.id())

for gid in gids_sca_first_node:
    pc.set_gid2node(gid, pc.id())

for gid in gids_sca_last_node:
    pc.set_gid2node(gid, pc.id())

if rank == 0:  
    print('[+] Creating cells')
    sys.stdout.flush()

ca1_pyr_cells = []
for gid_soma, gid_axon in zip(gids_pyr_soma, gids_pyr_axon):
    cell_ = PyramidalCell(gid_soma=gid_soma, gid_axon=gid_axon, 
                          x=ca1_pyr_coords[int(gid_soma/2), 0], y=ca1_pyr_coords[int(gid_soma/2), 1], theta=ca1_pyr_coords[int(gid_soma/2), 2],
                          x_intrinsic=ca1_pyr_coords[int(gid_soma/2), 3], y_intrinsic=ca1_pyr_coords[int(gid_soma/2), 4],
                          x_flat=ca1_pyr_coords[int(gid_soma/2), 5], y_flat=ca1_pyr_coords[int(gid_soma/2), 6])
    ca1_pyr_cells.append(cell_)
    # associate gid to spike_detector
    pc.cell(gid_soma, cell_._spike_detector)
    pc.cell(gid_axon, cell_._spike_detector_axon)

ca1_bc_cells = []
for gid in gids_bc:
    cell_ = BasketCell(gid=gid, 
                       x=ca1_bc_coords[gid - 2*n_pyr_ca1, 0], y=ca1_bc_coords[gid - 2*n_pyr_ca1, 1], theta=ca1_bc_coords[gid - 2*n_pyr_ca1, 2],
                       x_intrinsic=ca1_bc_coords[gid - 2*n_pyr_ca1, 3], y_intrinsic=ca1_bc_coords[gid - 2*n_pyr_ca1, 4],
                       x_flat=ca1_bc_coords[gid - 2*n_pyr_ca1, 5], y_flat=ca1_bc_coords[gid - 2*n_pyr_ca1, 6])
    ca1_bc_cells.append(cell_)
    # associate gid to spike_detector
    pc.cell(gid, cell_._spike_detector)

ca1_olm_cells = []
for gid in gids_olm:
    cell_ = OLMCell(gid=gid, 
                    x=ca1_olm_coords[gid - 2*n_pyr_ca1 - n_bc_ca1, 0], y=ca1_olm_coords[gid - 2*n_pyr_ca1 - n_bc_ca1, 1], theta=ca1_olm_coords[gid - 2*n_pyr_ca1 - n_bc_ca1, 2],
                    x_intrinsic=ca1_olm_coords[gid - 2*n_pyr_ca1 - n_bc_ca1, 3], y_intrinsic=ca1_olm_coords[gid - 2*n_pyr_ca1 - n_bc_ca1, 4],
                    x_flat=ca1_olm_coords[gid - 2*n_pyr_ca1 - n_bc_ca1, 5], y_flat=ca1_olm_coords[gid - 2*n_pyr_ca1 - n_bc_ca1, 6])
    ca1_olm_cells.append(cell_)
    # associate gid to spike_detector
    pc.cell(gid, cell_._spike_detector)

ca1_cells = ca1_pyr_cells + ca1_bc_cells + ca1_olm_cells

ca3_schaffers = []
for gid_first_node, gid_last_node in zip(gids_sca_first_node, gids_sca_last_node):
    cell_ = SchafferCollateral_2(gid=gid_first_node, gid_last_node=gid_last_node, nodes_coordinates=ca3_schaffer_nodes_coordinates[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2)],
                          x_intrinsic_last_node=ca3_schaffer_xs_intrinsic[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2)], y_intrinsic_last_node=ca3_schaffer_ys_intrinsic[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2)],
                          x_flat_last_node=ca3_schaffer_xs_flat[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2)], y_flat_last_node=ca3_schaffer_ys_flat[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2)],
                          x=ca3_schaffer_nodes_coordinates[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2)][0, 0], y=ca3_schaffer_nodes_coordinates[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2)][0, 1]
                          )
    ca3_schaffers.append(cell_)
    # associate gid to spike_detector
    pc.cell(gid_first_node, cell_._spike_detector)
    pc.cell(gid_last_node, cell_._spike_detector_last_node)

all_cells = ca1_cells + ca3_schaffers

# add noise
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

    print('[+] Finding all postsynaptic cells...')
    sys.stdout.flush()

# set connection matrix 
conn_mat = np.zeros((n_all_cells, n_all_cells))

# using flattened coordinates
for i in range(n_pyr_ca1):
    for j in range(n_bc_ca1):
        dist_value = np.sqrt((ca1_pyr_coords[i, 5] - ca1_bc_coords[j, 5])**2 + (ca1_pyr_coords[i, 6] - ca1_bc_coords[j, 6])**2)
        if settings.syn_dist_CA1[0] >= dist_value:
            conn_mat[i, j + n_pyr_ca1] = 1

    for j in range(n_olm_ca1):
        dist_value = np.sqrt((ca1_pyr_coords[i, 5] - ca1_olm_coords[j, 5])**2 + (ca1_pyr_coords[i, 6] - ca1_olm_coords[j, 6])**2)
        if settings.syn_dist_CA1[0] >= dist_value:
            conn_mat[i, j + n_pyr_ca1 + n_bc_ca1] = 1

for i in range(n_bc_ca1):
    for j in range(n_pyr_ca1):
        dist_value = np.sqrt((ca1_bc_coords[i, 5] - ca1_pyr_coords[j, 5])**2 + (ca1_bc_coords[i, 6] - ca1_pyr_coords[j, 6])**2)
        if settings.syn_dist_CA1[1] >= dist_value and conn_mat[n_pyr_ca1:n_pyr_ca1+n_bc_ca1, j].sum(axis=0) < 1: # only one conn. from BC to Pyr
            conn_mat[i + n_pyr_ca1, j] = 1

    for j in range(n_bc_ca1):
        dist_value = np.sqrt((ca1_bc_coords[i, 5] - ca1_bc_coords[j, 5])**2 + (ca1_bc_coords[i, 6] - ca1_bc_coords[j, 6])**2)
        if settings.syn_dist_CA1[1] >= dist_value and i != j:
            conn_mat[i + n_pyr_ca1, j + n_pyr_ca1] = 1

for i in range(n_olm_ca1):
    for j in range(n_pyr_ca1):
        dist_value = np.sqrt((ca1_olm_coords[i, 5] - ca1_pyr_coords[j, 5])**2 + (ca1_olm_coords[i, 6] - ca1_pyr_coords[j, 6])**2)
        if settings.syn_dist_CA1[2] >= dist_value:
            conn_mat[i + n_pyr_ca1 + n_bc_ca1, j] = 1

# connection from CA3
for i in range(len(ca3_schaffer_xs_flat)):
    for j in range(len(ca1_pyr_coords)):
        dist_value = np.sqrt((ca3_schaffer_xs_flat[i] - ca1_pyr_coords[j, 5])**2 + (ca3_schaffer_ys_flat[i] - ca1_pyr_coords[j, 6])**2)
        if settings.syn_dist_CA3_CA1[0] >= dist_value: # and conn_mat[n_cells_ca1: , j].sum(axis=0) < 3: # 3 schaffer inputs for each pyramidal cell
            conn_mat[n_cells_ca1 + i, j] = 1

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

for i in range(n_schaffers_ca3):
    for j in range(n_pyr_ca1):
        if conn_mat[n_cells_ca1 + i, j] > 0 and pc.gid_exists(n_cells_ca1 + n_pyr_ca1 + 2*i + 1): # connect to last node of schaffer collaterals
            pc.gid2cell(n_cells_ca1 + n_pyr_ca1 + 2*i + 1)._postsyn_list.append(2*j)
        if conn_mat[n_cells_ca1 + i, j] > 0 and pc.gid_exists(2*j):
            pc.gid2cell(2*j)._presyn_list.append(n_cells_ca1 + n_pyr_ca1 + 2*i + 1)


if rank == 0:
    print('[+] Connecting cells...')
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
            nc_.weight[0] = settings.w_CA1[1][0] 
            nc_.threshold = settings.syn_threshold
            nc_.delay = settings.syn_delay
            cell_._ncs.append(nc_)
        elif pregid >= 2*n_pyr_ca1 + n_bc_ca1 and pregid < n_pyr_ca1 + n_cells_ca1: # from OLM
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
        else: # from schaffer collaterals
            target_secs = list(cell_.rad_list)
            target_sec = random.choice(target_secs)
            mt_ = h.MechanismType(1)
            mt_.select("Exp2Syn")
            pp = mt_.pp_begin(sec=target_sec)
            nc_ = pc.gid_connect(pregid, pp)
            nc_.weight[0] = settings.w_CA3_CA1[0] 
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
            nc_.weight[0] = settings.w_CA1[0][1] 
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

pc.barrier()

# create inputs
if rank == 0:
    print('[+] Connecting cells done')
    print('\n[20] Setting the inputs...')
    sys.stdout.flush()

if rank == 0:
    print('[+] Oscillatory input')
    sys.stdout.flush()

for cell_ in ca3_schaffers:
    r_osc = random.uniform(0, 50)
    target_sec = cell_.soma
    input_ = oscInput(cell_, target_sec(0.5), settings.input_onset+r_osc, settings.input_duration, settings.input_rate, settings.input_amp, noisy=False)
    cell_._inputs_vector.record(input_._ref_i)

pc.barrier()

if rank == 0:
    print('[+] Inputs done')
    print('\n[30] Stimulation...')
    print('-'*32)

# set stim parameters
stim_amp = h.Vector()
stim_time = h.Vector()

if settings.stim_type == "train_pulses":
    stim_amp, stim_time = train_pulse(stim_amp, stim_time, settings.stim_amp, settings.stim_onset, settings.stim_dur, settings.stim_pulse_width, settings.stim_freq, settings.stim_waveform, settings.duration, settings.stim_interphase) 
elif settings.stim_type == "theta_burst":
    stim_amp, stim_time = theta_burst(stim_amp=stim_amp, stim_time=stim_time, amp=settings.stim_amp, onset=settings.stim_onset, duration=settings.stim_dur,
                                      pulse_width=settings.stim_pulse_width, n_pulses=settings.stim_n_pulses, theta_frequency=settings.stim_theta_frequency,
                                      frequency=settings.stim_freq, stim_type=settings.stim_waveform, sim_dur=settings.duration, interphase=settings.stim_interphase)
else:
    stim_amp, stim_time = single_pulse(stim_amp, stim_time, settings.stim_onset, settings.stim_dur, settings.stim_amp, settings.duration)


# Set extracellular stimulation
# set xtra mechanism in all cells
for cell in all_cells:
    set_xtra_mechanism(cell)
    if settings.stim_electrode == "bipolar":
        set_rx_bipolar(cell, settings.stim_pos[0], settings.stim_pos[1], settings.rho)
    else:
        set_rx_point_elec(cell, settings.stim_pos, settings.rho)
    attach_stim(cell, settings.ATTACHED__, stim_amp, stim_time)

if rank == 0:
    print('[+] Extracellular mechanism set')
    print('\n[40] Simulation...')
    print('-'*32)

    print('[+] Setting recording vectors')
    sys.stdout.flush()

# Set recording vectors
t_vec = h.Vector().record(h._ref_t)

# set finitializehandler
fih = h.FInitializeHandler(1, _set_v_init) # random initialization of Vm

h.tstop = settings.duration 
h.celsius = settings.sim_celsius

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
local_potential_sca = {cell_._gid: np.array(cell_.soma_v) for cell_ in ca3_schaffers}
local_potential_sca_last = {cell_._gid: np.array(cell_.last_node_v) for cell_ in ca3_schaffers}

# local spike times
local_spikes_pyr = {cell_._gid: list(cell_.spike_times) for cell_ in ca1_pyr_cells}
local_spikes_bc = {cell_._gid: list(cell_.spike_times) for cell_ in ca1_bc_cells}
local_spikes_olm = {cell_._gid: list(cell_.spike_times) for cell_ in ca1_olm_cells}
local_spikes_sca = {cell_._gid: list(cell_.spike_times) for cell_ in ca3_schaffers}
local_spikes_sca_last = {cell_._gid: list(cell_.spike_times_last_node) for cell_ in ca3_schaffers}

# local inputs
local_inputs_sca = {cell_._gid: list(cell_._inputs_vector) for cell_ in ca3_schaffers}

# send results to all processors
all_potential_pyr = pc.py_alltoall([local_potential_pyr] + [None] * (pc.nhost() - 1))
all_potential_bc = pc.py_alltoall([local_potential_bc] + [None] * (pc.nhost() - 1))
all_potential_olm = pc.py_alltoall([local_potential_olm] + [None] * (pc.nhost() - 1))
all_potential_sca = pc.py_alltoall([local_potential_sca] + [None] * (pc.nhost() - 1))
all_potential_sca_last = pc.py_alltoall([local_potential_sca_last] + [None] * (pc.nhost() - 1))

all_spikes_pyr = pc.py_alltoall([local_spikes_pyr] + [None] * (pc.nhost() - 1))
all_spikes_bc = pc.py_alltoall([local_spikes_bc] + [None] * (pc.nhost() - 1))
all_spikes_olm = pc.py_alltoall([local_spikes_olm] + [None] * (pc.nhost() - 1))
all_spikes_sca = pc.py_alltoall([local_spikes_sca] + [None] * (pc.nhost() - 1))
all_spikes_sca_last = pc.py_alltoall([local_spikes_sca_last] + [None] * (pc.nhost() - 1))

all_inputs_sca = pc.py_alltoall([local_inputs_sca] + [None] * (pc.nhost() - 1))

if rank == 0:
    # combine the data from the various processors
    potential_pyr = {}
    potential_bc = {}
    potential_olm = {}
    potential_sca = {}
    potential_sca_last = {}

    spikes_pyr = {}
    spikes_bc = {}
    spikes_olm = {}
    spikes_sca = {}
    spikes_sca_last = {}

    inputs_sca = {}

    for data in all_potential_pyr:
        potential_pyr.update(data)

    for data in all_potential_bc:
        potential_bc.update(data)

    for data in all_potential_olm:
        potential_olm.update(data)

    for data in all_potential_sca:
        potential_sca.update(data)

    for data in all_potential_sca_last:
        potential_sca_last.update(data)

    
    for data in all_spikes_pyr:
        spikes_pyr.update(data)

    for data in all_spikes_bc:
        spikes_bc.update(data)

    for data in all_spikes_olm:
        spikes_olm.update(data)

    for data in all_spikes_sca:
        spikes_sca.update(data)

    for data in all_spikes_sca_last:
        spikes_sca_last.update(data)


    for data in all_inputs_sca:
        inputs_sca.update(data)

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

    id_spikes_sca = []
    t_spikes_sca = []
    for key, value in spikes_sca.items():
        id_spikes_sca.extend([n_cells_ca1 + int(key-n_pyr_ca1-n_cells_ca1)/2] * len(value))
        t_spikes_sca.extend(value)

    id_spikes_sca_last = []
    t_spikes_sca_last = []
    for key, value in spikes_sca_last.items():
        id_spikes_sca_last.extend([n_cells_ca1 + int(key-n_pyr_ca1-n_cells_ca1)/2] * len(value))
        t_spikes_sca_last.extend(value)

    with open(os.path.join(dirs["save_dir"], "output.txt"), "w") as f:
        f.write("Git parameters\n")
        f.write("-------------------\n")
        for key, value in git_kwargs.items():
            f.write(f"{key} : {value}\n")
        
        f.close()

    # save vectors
    save_membrane_potential(os.path.join(dirs['data'], 'CA1_pyr_Vm.npz'), np.array(t_vec), potential_pyr) 
    save_membrane_potential(os.path.join(dirs['data'], 'CA1_bc_Vm.npz'), np.array(t_vec), potential_bc) 
    save_membrane_potential(os.path.join(dirs['data'], 'CA1_olm_Vm.npz'), np.array(t_vec), potential_olm) 
    save_membrane_potential(os.path.join(dirs['data'], 'CA3_sca_Vm.npz'), np.array(t_vec), potential_sca) 
    save_membrane_potential(os.path.join(dirs['data'], 'CA3_sca_last_Vm.npz'), np.array(t_vec), potential_sca_last)

    save_membrane_potential(os.path.join(dirs['data'], 'theta_inputs.npz'), np.array(t_vec), inputs_sca)  

    np.savez(os.path.join(dirs['data'], 'CA1_pyr_spikemon.npz'), cell_id=np.array(id_spikes_pyr), t_spike=np.array(t_spikes_pyr))
    np.savez(os.path.join(dirs['data'], 'CA1_bc_spikemon.npz'), cell_id=np.array(id_spikes_bc), t_spike=np.array(t_spikes_bc))
    np.savez(os.path.join(dirs['data'], 'CA1_olm_spikemon.npz'), cell_id=np.array(id_spikes_olm), t_spike=np.array(t_spikes_olm))
    np.savez(os.path.join(dirs['data'], 'CA3_sca_spikemon.npz'), cell_id=np.array(id_spikes_sca), t_spike=np.array(t_spikes_sca))
    np.savez(os.path.join(dirs['data'], 'CA3_sca_last_spikemon.npz'), cell_id=np.array(id_spikes_sca_last), t_spike=np.array(t_spikes_sca_last))

    np.savez(os.path.join(dirs['data'], 'i_noise.npz'), t=np.array(t_vec), noise=np.array(i_noise))
    np.savez(os.path.join(dirs['data'], 'stimulation.npz'), t=np.array(stim_time), amp=np.array(stim_amp))

    print("[+] Saving data done !")
    print("[+] Resizing vectors")
    sys.stdout.flush()

# wait for all processors to reach this stage
pc.barrier()
pc.done()
h.quit()
