import os
import sys
import time
from datetime import datetime
import argparse

from neuron import h
from Cells.Cells import PyramidalCell, BasketCell, OLMCell, SchafferCollateral, SchafferCollateral_2
from Scripts.Stimulation import *
from Scripts.Network import *
from Scripts.myplot import save_raster, save_FR, save_specgram, plot_watermark
from Scripts.utilities import *
from Scripts.anatomy import *
from Scripts.input import *
from Model import settings
import parameters
import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
import random
from collections import OrderedDict

# sys.exit()

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

def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.
	from : https://stackoverflow.com/a/35501485

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range

def plot_2D_custom(h, ax, section, colors, d=0):
	'''
		Parameters:
			- d : additional diameter -> increase diameter of section by value of d. Can be used for hovering
	'''
	n3d = int(h.n3d(sec=section))
	xyd = []
	for i in range(0,n3d):
		xyd.append([h.x3d(i,sec=section),h.y3d(i,sec=section),h.diam3d(i,sec=section)+d])
	xyd = np.array(xyd)
	if len(colors) > 1 and len(colors) == len(xyd)-1:
		#assert len(colors) == len(xyd)-1, (f"colors length must be 1 or {len(xyd)-1} not {len(colors)}")
		for i in range(len(xyd) - 2):
			ax.plot(xyd[i:i+2,0], xyd[i:i+2,1], color=colors[i], linewidth=linewidth_from_data_units(int(xyd[0,2]), ax))
		ax.plot(xyd[-2:,0], xyd[-2:,1], color=colors[-1], linewidth=linewidth_from_data_units(int(xyd[0,2]), ax))
	else:
		ax.plot(xyd[:,0], xyd[:,1], color=colors[0], linewidth=linewidth_from_data_units(int(xyd[0,2]), ax))
          

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
parser = argparse.ArgumentParser(description='Multicomp model with schaffer collaterals parameters search')

parser.add_argument('-p', '--parameters',
                    nargs='?',
                    metavar='-p',
                    type=str,
                    default=os.path.join('configs', 'parameters_no_stim.json'), #TODO: change parameters file
                    help='Parameters file (json format)')

parser.add_argument('-sd', '--save_dir',
                    nargs='?',
                    metavar='-sd',
                    type=str,
                    default='network_with_schaffers_results',
                    help='Destination directory to save the results')

args = parser.parse_args()
filename = args.parameters
resdir = args.save_dir

try:
    data = parameters.load(filename)
    print('Using "{0}"'.format(filename))
except Exception as e:
    print(e)
    print('Using "parameters_no_stim.json"')
    data = parameters._data
parameters.dump(data) # TODO: update file after changing weights ?
print()

# Settings initialization
settings.init(data)

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

dirs['save_dir'] = os.path.join(dirs['results'], datetime.now().strftime(f"%Y_%m_%d %HH%MM%S no stim - new schaffer distribution McIntyre model"))
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
    print('\n[10] Making the neuron populations...')
    print('-'*32)

    print('[+] Loading coordinates')
    sys.stdout.flush()


# retrieve coordinates
# coordinates_dir = 'Positions'
coordinates_dir = 'positions_correct_layers_thickness'
ca1_coordinates = os.path.join(coordinates_dir, 'ca1')
ca3_coordinates = os.path.join(coordinates_dir, 'ca3')

ca1_pyr_coords = np.load(os.path.join(ca1_coordinates, 'pyr_coordinates.npy'))
ca1_bc_coords = np.load(os.path.join(ca1_coordinates, 'bc_coordinates.npy'))
ca1_olm_coords = np.load(os.path.join(ca1_coordinates, 'olm_coordinates.npy'))

ca3_schaffer_coords = np.load(os.path.join(ca3_coordinates, 'pyr_coordinates.npy'))
ca3_schaffer_nodes_coordinates = np.load(os.path.join(ca3_coordinates, 'nodes_coordinates.npy'), allow_pickle=True)
ca3_schaffer_xs_intrinsic = np.load(os.path.join(ca3_coordinates, 'xs_intrinsic_last_node.npy'), allow_pickle=True)
ca3_schaffer_ys_intrinsic = np.load(os.path.join(ca3_coordinates, 'ys_intrinsic_last_node.npy'), allow_pickle=True)
ca3_schaffer_xs_flat = np.load(os.path.join(ca3_coordinates, 'xs_flat_last_node.npy'), allow_pickle=True)
ca3_schaffer_ys_flat = np.load(os.path.join(ca3_coordinates, 'ys_flat_last_node.npy'), allow_pickle=True)
# ca3_schaffer_trajectories = np.load(os.path.join(ca3_coordinates, 'schaffer_trajectories_all_coords.npy'), allow_pickle=True)


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

print('gids_pyr_soma : ', gids_pyr_soma)
print('gids_pyr_axon : ', gids_pyr_axon)

gids_interneurons = list(range(pc.id() + 2*n_pyr_ca1, n_cells_ca1+n_pyr_ca1, pc.nhost()))

print('gids_interneurons : ', gids_interneurons)

gids_bc = [gid for gid in gids_interneurons if gid < 2*n_pyr_ca1 + n_bc_ca1]

print('gids_bc : ', gids_bc)

gids_olm = [gid for gid in gids_interneurons if gid >= 2*n_pyr_ca1 + n_bc_ca1]

print('gids_olm : ', gids_olm)

gids_sca_first_node = [n_cells_ca1+n_pyr_ca1+2*n for n in range(pc.id(), n_schaffers_ca3, pc.nhost())]
gids_sca_last_node = [n_cells_ca1+n_pyr_ca1+2*n + 1 for n in range(pc.id(), n_schaffers_ca3, pc.nhost())]

print('gids_sca_first_node : ', gids_sca_first_node)
print('gids_sca_last_node : ', gids_sca_last_node)

# sys.exit()

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
                          x=ca3_schaffer_coords[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2), 0], y=ca3_schaffer_coords[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2), 1], theta=ca3_schaffer_coords[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2), 2],
                          x_intrinsic=ca3_schaffer_coords[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2), 3], y_intrinsic=ca3_schaffer_coords[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2), 4],
                          x_flat=ca3_schaffer_coords[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2), 5], y_flat=ca3_schaffer_coords[int((gid_first_node-n_cells_ca1-n_pyr_ca1)/2), 6])
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

# for cell_ in ca3_schaffers: #TODO: add to parameters file + in upper loop
#     for sec in cell_.all:
#         sec.insert("Inoise")
#         sec.myseed_Inoise = RNG.integers(0, 1000000000000000, 1) #seed_noise #settings.seed_noise
#         if "PyramidalCell" in str(cell_):
#             sec.sigma_Inoise = 0.00000001
#             sec.mean_Inoise = 0
#             sec.tau_Inoise = 5
#             sec.inoise_Inoise = -1.0

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

# set connection matrix #TODO: continue from here
conn_mat = np.zeros((n_all_cells, n_all_cells))

# using flattened coordinates
for i in range(n_pyr_ca1):
    # for j in range(n_pyr_ca1):
    #     dist_value = np.sqrt((ca1_pyr_coords[i, 5] - ca1_pyr_coords[j, 5])**2 + (ca1_pyr_coords[i, 6] - ca1_pyr_coords[j, 6])**2)
    #     if settings.syn_dist_CA1[0] >= dist_value and i != j and conn_mat[i, :n_pyr_ca1].sum(axis=0) < 1 and conn_mat[:n_pyr_ca1, j].sum(axis=0) < 1:
    #         conn_mat[i, j] = 1

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

np.save(os.path.join(dirs['data'], "connection_matrix.npy"), conn_mat)

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
    print('[+] Found all postsynatpic cells')
    print('[+] Computing number of connections per cell')
    sys.stdout.flush()

# postsynaptic conns/divergence

pyr_to_pyr = 0
pyr_to_bc = 0
pyr_to_olm = 0

bc_to_pyr = 0
bc_to_bc = 0
bc_to_olm = 0

olm_to_pyr = 0
olm_to_bc = 0
olm_to_olm = 0

sca_to_pyr = 0

for cell_ in ca1_pyr_cells:
    for gid in cell_._postsyn_list:
        if gid < n_pyr_ca1 * 2:
            pyr_to_pyr += 1
        elif gid >= n_pyr_ca1 * 2 and gid < n_pyr_ca1 * 2 + n_bc_ca1:
            pyr_to_bc += 1
        else:
            pyr_to_olm += 1

for cell_ in ca1_bc_cells:
    for gid in cell_._postsyn_list:
        if gid < n_pyr_ca1 * 2:
            bc_to_pyr += 1
        elif gid >= n_pyr_ca1 * 2 and gid < n_pyr_ca1 * 2 + n_bc_ca1:
            bc_to_bc += 1
        else:
            bc_to_olm += 1

for cell_ in ca1_olm_cells:
    for gid in cell_._postsyn_list:
        if gid < n_pyr_ca1 * 2:
            olm_to_pyr += 1
        elif gid >= n_pyr_ca1 * 2 and gid < n_pyr_ca1 * 2 + n_bc_ca1:
            olm_to_bc += 1
        else:
            olm_to_olm += 1

for cell_ in ca3_schaffers:
    for gid in cell_._postsyn_list:
        if gid < n_pyr_ca1 * 2:
            sca_to_pyr += 1

local_pyr_post_conn = {rank: [pyr_to_pyr, pyr_to_bc, pyr_to_olm]}
local_bc_post_conn = {rank: [bc_to_pyr, bc_to_bc, bc_to_olm]}
local_olm_post_conn = {rank: [olm_to_pyr, olm_to_bc, olm_to_olm]}
local_sca_post_conn = {rank: [sca_to_pyr]}

# presynaptic conns/convergence
pyr_from_pyr = 0
pyr_from_bc = 0
pyr_from_olm = 0

bc_from_pyr = 0
bc_from_bc = 0
bc_from_olm = 0

olm_from_pyr = 0
olm_from_bc = 0
olm_from_olm = 0

for cell_ in ca1_pyr_cells:
    for gid in cell_._presyn_list:
        if gid < n_pyr_ca1 * 2:
            pyr_from_pyr += 1
        elif gid >= n_pyr_ca1 * 2 and gid < n_pyr_ca1 * 2 + n_bc_ca1:
            pyr_from_bc += 1
        else:
            pyr_from_olm += 1

for cell_ in ca1_bc_cells:
    for gid in cell_._presyn_list:
        if gid < n_pyr_ca1 * 2:
            bc_from_pyr += 1
        elif gid >= n_pyr_ca1 * 2 and gid < n_pyr_ca1 * 2 + n_bc_ca1:
            bc_from_bc += 1
        else:
            bc_from_olm += 1

for cell_ in ca1_olm_cells:
    for gid in cell_._presyn_list:
        if gid < n_pyr_ca1 * 2:
            olm_from_pyr += 1
        elif gid >= n_pyr_ca1 * 2 and gid < n_pyr_ca1 * 2 + n_bc_ca1:
            olm_from_bc += 1
        else:
            olm_from_olm += 1

local_pyr_pre_conn = {rank: [pyr_from_pyr, pyr_from_bc, pyr_from_olm]}
local_bc_pre_conn = {rank: [bc_from_pyr, bc_from_bc, bc_from_olm]}
local_olm_pre_conn = {rank: [olm_from_pyr, olm_from_bc, olm_from_olm]}

# send to all
all_pyr_post_conn = pc.py_alltoall([local_pyr_post_conn] + [None] * (pc.nhost() - 1))
all_bc_post_conn = pc.py_alltoall([local_bc_post_conn] + [None] * (pc.nhost() - 1))
all_olm_post_conn = pc.py_alltoall([local_olm_post_conn] + [None] * (pc.nhost() - 1))
all_sca_post_conn = pc.py_alltoall([local_sca_post_conn] + [None] * (pc.nhost() - 1))

all_pyr_pre_conn = pc.py_alltoall([local_pyr_pre_conn] + [None] * (pc.nhost() - 1))
all_bc_pre_conn = pc.py_alltoall([local_bc_pre_conn] + [None] * (pc.nhost() - 1))
all_olm_pre_conn = pc.py_alltoall([local_olm_pre_conn] + [None] * (pc.nhost() - 1))

if rank == 0:
    # combine the data from the various processes
    pyr_post_conn = {}
    bc_post_conn = {}
    olm_post_conn = {}
    sca_post_conn = {}

    pyr_pre_conn = {}
    bc_pre_conn = {}
    olm_pre_conn = {}

    for data in all_pyr_post_conn:
        pyr_post_conn.update(data)

    for data in all_bc_post_conn:
        bc_post_conn.update(data)

    for data in all_olm_post_conn:
        olm_post_conn.update(data)

    for data in all_sca_post_conn:
        sca_post_conn.update(data)

    for data in all_pyr_pre_conn:
        pyr_pre_conn.update(data)
    
    for data in all_bc_pre_conn:
        bc_pre_conn.update(data)

    for data in all_olm_pre_conn:
        olm_pre_conn.update(data)

    global_pyr_to_pyr = 0
    global_pyr_to_bc = 0
    global_pyr_to_olm = 0

    global_bc_to_pyr = 0
    global_bc_to_bc = 0
    global_bc_to_olm = 0

    global_olm_to_pyr = 0
    global_olm_to_bc = 0
    global_olm_to_olm = 0

    global_sca_to_pyr = 0

    global_pyr_from_pyr = 0
    global_pyr_from_bc = 0
    global_pyr_from_olm = 0

    global_bc_from_pyr = 0
    global_bc_from_bc = 0
    global_bc_from_olm = 0

    global_olm_from_pyr = 0
    global_olm_from_bc = 0
    global_olm_from_olm = 0

    for values in pyr_post_conn.values():
        global_pyr_to_pyr += values[0]
        global_pyr_to_bc += values[1]
        global_pyr_to_olm += values[2]

    for values in bc_post_conn.values():
        global_bc_to_pyr += values[0]
        global_bc_to_bc += values[1]
        global_bc_to_olm += values[2]

    for values in olm_post_conn.values():
        global_olm_to_pyr += values[0]
        global_olm_to_bc += values[1]
        global_olm_to_olm += values[2]

    for values in sca_post_conn.values():
        global_sca_to_pyr += values[0]

    for values in pyr_pre_conn.values():
        global_pyr_from_pyr += values[0]
        global_pyr_from_bc += values[1]
        global_pyr_from_olm += values[2]

    for values in bc_pre_conn.values():
        global_bc_from_pyr += values[0]
        global_bc_from_bc += values[1]
        global_bc_from_olm += values[2]

    for values in olm_pre_conn.values():
        global_olm_from_pyr += values[0]
        global_olm_from_bc += values[1]
        global_olm_from_olm += values[2]

    print('-'*32)
    print("Mean number of connections")
    print("pyr to pyr : {}".format(global_pyr_to_pyr/n_pyr_ca1))
    print("pyr to bc : {}".format(global_pyr_to_bc/n_pyr_ca1))
    print("pyr to olm : {}\n".format(global_pyr_to_olm/n_pyr_ca1))

    print("bc to pyr : {}".format(global_bc_to_pyr/n_bc_ca1))
    print("bc to bc : {}".format(global_bc_to_bc/n_bc_ca1))
    print("bc to olm : {}\n".format(global_bc_to_olm/n_bc_ca1))

    print("olm to pyr : {}".format(global_olm_to_pyr/n_olm_ca1))
    print("olm to bc : {}".format(global_olm_to_bc/n_olm_ca1))
    print("olm to olm : {}".format(global_olm_to_olm/n_olm_ca1))

    print("sca to pyr : {}".format(global_sca_to_pyr/n_schaffers_ca3))
    print('*'*32)

    print("pyr from pyr : {}".format(global_pyr_from_pyr/n_pyr_ca1))
    print("pyr from bc : {}".format(global_pyr_from_bc/n_pyr_ca1))
    print("pyr from olm : {}\n".format(global_pyr_from_olm/n_pyr_ca1))

    print("bc from pyr : {}".format(global_bc_from_pyr/n_bc_ca1))
    print("bc from bc : {}".format(global_bc_from_bc/n_bc_ca1))
    print("bc from olm : {}\n".format(global_bc_from_olm/n_bc_ca1))

    print("olm from pyr : {}".format(global_olm_from_pyr/n_olm_ca1))
    print("olm from bc : {}".format(global_olm_from_bc/n_olm_ca1))
    print("olm from olm : {}".format(global_olm_from_olm/n_olm_ca1))
    print('-'*32)
    sys.stdout.flush()

pc.barrier()
# sys.exit()
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
            nc_.weight[0] = settings.w_CA3_CA1[0]*3
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
            nc_.weight[0] = settings.w_CA1[0][2] # try with same as BC
            nc_.threshold = settings.syn_threshold
            nc_.delay = settings.syn_delay
            cell_._ncs.append(nc_)

pc.barrier()

# create inputs
# inputs are set as intracellular ramp current to soma of pyramidal and basket cells
if rank == 0:
    print('[+] Connecting cells done')
    print('\n[20] Setting the inputs...')
    sys.stdout.flush()

if rank == 0:
    print('[+] Oscillatory input')
    sys.stdout.flush()

duration = 5000
amp = 0.05#0.005 #0 #6.0
delay = 1000

# osc_amp = h.Vector()
# setting oscillatory input current
for cell_ in ca3_schaffers:
    # target_secs = list(cell_.lm_list)
    # target_sec = random.choice(target_secs)
    r_osc = random.uniform(0, 50)
    # r_osc = 0
    target_sec = cell_.soma
    input_ = oscInput(cell_, target_sec(0.5), delay+r_osc, duration, 6, amp, noisy=False)
    # input_ = oscInput(cell_, target_sec(0.5), 10, duration, 6, amp, noisy=True, sigma=0.005, tau_noise=1)
    # osc_amp.record(input_._ref_i)
    # cell_._inputs_list.append(osc_amp)
    cell_._inputs_vector.record(input_._ref_i)

# setting current vectors
# VecT = h.Vector([0, TSTOP])
# VecStim = h.Vector([0., 5.])
# stim_amp = h.Vector()

# for cell_ in ca1_cells:
#     stim_ = h.IClamp(cell_.soma(0.5))
#     stim_.delay = 0
#     stim_.dur = 1e9

#     VecStim.play(stim_._ref_amp, VecT, 1)
#     cell_._inputs_list.append(stim_)

# for cell_ in ca1_cells:
#     if 'PyramidalCell' in str(cell_):
#         stim_ = h.IClamp(cell_.soma(0.5))
#         stim_.delay = 0
#         stim_.dur = 1e9

#         VecStim.play(stim_._ref_amp, VecT, 1)
#         cell_._inputs_list.append(stim_)

#         stim_amp.record(stim_._ref_i) # last stim_

pc.barrier()

if rank == 0:
    print('[+] Inputs done')
    print('\n[30] Stimulation...')
    print('-'*32)

# set stim parameters
# stim_amp = h.Vector()
# stim_time = h.Vector()

# stim_amp, stim_time = stim_waveform(stim_amp, stim_time, settings.stim_onset, settings.stim_dur, settings.stim_amp)

# Set extracellular stimulation
# set xtra mechanism in all cells
# for cell in ca1_cells:
#     set_xtra_mechanism(cell)
#     if settings.stim_type == "bipolar":
#         set_rx_bipolar(cell, settings.stim_pos[0], settings.stim_pos[1], settings.rho)
#     else:
#         set_rx_point_elec(cell, settings.stim_pos, settings.rho)
#     attach_stim(cell, settings.ATTACHED__, stim_amp, stim_time)

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

h.tstop = duration #settings.duration
h.celsius = settings.sim_celsius

pc.set_maxstep(10)

if rank == 0:
    print('[+] Running simulation...')
    sys.stdout.flush()

h.stdinit()
h.cvode_active(0)
start_time = time.time()
pc.psolve(duration)
# pc.psolve(settings.duration)
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

# local intracellular_current
local_current_pyr = {cell_._gid: np.array(cell_.soma_i) for cell_ in ca1_pyr_cells}
local_current_bc = {cell_._gid: np.array(cell_.soma_i) for cell_ in ca1_bc_cells}
local_current_olm = {cell_._gid: np.array(cell_.soma_i) for cell_ in ca1_olm_cells}
local_current_sca = {cell_._gid: np.array(cell_.soma_i) for cell_ in ca3_schaffers}
local_current_sca_last = {cell_._gid: np.array(cell_.last_node_i) for cell_ in ca3_schaffers}

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

all_current_pyr = pc.py_alltoall([local_current_pyr] + [None] * (pc.nhost() - 1))
all_current_bc = pc.py_alltoall([local_current_bc] + [None] * (pc.nhost() - 1))
all_current_olm = pc.py_alltoall([local_current_olm] + [None] * (pc.nhost() - 1))
all_current_sca = pc.py_alltoall([local_current_sca] + [None] * (pc.nhost() - 1))
all_current_sca_last = pc.py_alltoall([local_current_sca_last] + [None] * (pc.nhost() - 1))

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

    current_pyr = {}
    current_bc = {}
    current_olm = {}
    current_sca = {}
    current_sca_last = {}

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

    for data in all_current_pyr:
        current_pyr.update(data)

    for data in all_current_bc:
        current_bc.update(data)

    for data in all_current_olm:
        current_olm.update(data)

    for data in all_current_sca:
        current_sca.update(data)

    for data in all_current_sca_last:
        current_sca_last.update(data)

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

    # spectrogram
    cmesh_list = []
    vlow = []
    vhigh = []

    winsize_fr = 5 #ms
    overlap_fr = 0.9

    window_size = 1 #ms

    window_size_Pxx = 150 #1000
    window_width_Pxx = int(window_size_Pxx * (1/window_size))
    overlap_Pxx = 0.9
    window_overlap_Pxx = int(window_width_Pxx*overlap_Pxx)
    vmin = 1e-12
    vmax = 1.
    norm_bc = colors.Normalize(vmin=vmin, vmax=vmax)
    norm_olm = colors.Normalize(vmin=vmin, vmax=vmax)
    norm_pyr = colors.Normalize(vmin=vmin, vmax=vmax)
    norm_sca = colors.Normalize(vmin=vmin, vmax=vmax)

    specgram_kwargs = { 'return_onesided' : True,
                        'scaling' : 'density',
                        'mode' : 'magnitude' }

    # compute firing rates
    t_FR_pyr, count_pyr, FR_pyr, fs_n = compute_FR(np.array(t_spikes_pyr)*1e-3, n_pyr_ca1, duration*1e-3, winsize_fr*1e-3, overlap_fr)
    t_FR_bc, count_bc, FR_bc, _ = compute_FR(np.array(t_spikes_bc)*1e-3, n_bc_ca1, duration*1e-3, winsize_fr*1e-3, overlap_fr)
    t_FR_olm, count_olm, FR_olm, _ = compute_FR(np.array(t_spikes_olm)*1e-3, n_olm_ca1, duration*1e-3, winsize_fr*1e-3, overlap_fr)
    t_FR_sca, count_sca, FR_sca, _ = compute_FR(np.array(t_spikes_sca_last)*1e-3, n_schaffers_ca3, duration*1e-3, winsize_fr*1e-3, overlap_fr)

    with open(os.path.join(dirs["save_dir"], "output.txt"), "w") as f:
        f.write("Git parameters\n")
        f.write("-------------------\n")
        for key, value in git_kwargs.items():
            f.write(f"{key} : {value}\n")

        f.write("\n\nSimulation parameters\n")
        f.write("-------------------------\n")
        f.write("remark :\n")
        f.write("J'ai utilisé une nouvelle distribution des collatéraux de Schaffer\n")
        f.write("Comme ils ont l'air mieux distribués dans l'espace, j'ai enlevé la contrainte qui consister à avoir 3 connexions de Schaffer max par pyr")
        f.write("Je refais les expériences w_sca * 3 et offset (0, 50) pour un courant oscillatoire à 0.05nA")
        f.write("Collatéraux avec le modèle de McIntyre")
        f.write("- no Pyr - Pyr connections\n")
        f.write("- pyr-bc weights used for sca-pyr connections\n")
        f.write("- BC - Pyr constrained to one connection max\n")
        f.write("- Lamellar coordinates used\n")
        f.write("\nPyr - BC weight : {}\n".format(settings.w_CA1[0][1]))
        f.write("BC - Pyr weight : {}\n".format(settings.w_CA1[1][0]))
        f.write("BC - BC weight : {}\n".format(settings.w_CA1[1][1]))
        f.write("OLM - Pyr weight : {}\n".format(settings.w_CA1[2][0]))
        f.write("Pyr - OLM weight : {}\n".format(settings.w_CA1[0][2]))
        f.write("\nsca - pyr weight : {}\n".format(settings.w_CA3_CA1[0]*3))

        f.write("\n\nSimulation results\n")
        f.write("-------------------------\n")
        f.write("Mean firing rate over simulation\n")
        f.write("Pyr firing rate : {} Hz\n".format(np.mean(FR_pyr)))
        f.write("BC firing rate : {} Hz\n".format(np.mean(FR_bc)))
        f.write("OLM firing rate : {} Hz\n\n".format(np.mean(FR_olm)))
        f.write("Sca firing rate : {} Hz\n\n".format(np.mean(FR_sca)))

        f.write("Mean firing rate over last 1s\n")
        f.write("Pyr firing rate : {} Hz\n".format(np.mean(FR_pyr[-200:])))
        f.write("BC firing rate : {} Hz\n".format(np.mean(FR_bc[-200:])))
        f.write("OLM firing rate : {} Hz\n\n".format(np.mean(FR_olm[-200:])))
        f.write("Sca firing rate : {} Hz\n\n".format(np.mean(FR_sca[-200:])))

        # f.write("\n\nNodes details\n")
        # f.write("-------------------------\n")
        # f.write("Schaffer[0]\n")
        # f.write("N_nodes : {}\n".format(len(ca3_schaffers[0].nodes)))
        # f.write("Schaffer[2]\n")
        # f.write("N_nodes : {}\n".format(len(ca3_schaffers[2].nodes)))
        # f.write("Schaffer[19]\n")
        # f.write("N_nodes : {}\n".format(len(ca3_schaffers[19].nodes)))
        f.close()

    # compute spectrograms
    fv_pyr, tv_pyr, pspec_pyr = my_specgram2(FR_pyr, fs_n, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
    fv_bc, tv_bc, pspec_bc = my_specgram2(FR_bc, fs_n, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
    fv_olm, tv_olm, pspec_olm = my_specgram2(FR_olm, fs_n, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)
    fv_sca, tv_sca, pspec_sca = my_specgram2(FR_sca, fs_n, window_width_Pxx, window_overlap_Pxx, k=2, **specgram_kwargs)


    vlow.append(pspec_pyr.min())
    vhigh.append(pspec_pyr.max())

    for cmsh in make_flat(cmesh_list):
        cmsh.set_clim(min(vlow), max(vhigh))

    save_specgram(os.path.join(dirs['figures'], 'specgram.png'), [tv_pyr, tv_bc, tv_olm, tv_sca], [fv_pyr, fv_bc, fv_olm, fv_sca], [pspec_pyr, pspec_bc, pspec_olm, pspec_sca], ["pyramidal cells", "basket cells", "olm cells", "schaffer collaterals"], **git_kwargs)
    save_specgram(os.path.join(dirs['figures'], 'specgram_0_150.png'), [tv_pyr, tv_bc, tv_olm, tv_sca], [fv_pyr, fv_bc, fv_olm, fv_sca], [pspec_pyr, pspec_bc, pspec_olm, pspec_sca], ["pyramidal cells", "basket cells", "olm cells", "schaffer collaterals"], ylim=[0, 150], **git_kwargs)
    save_specgram(os.path.join(dirs['figures'], 'specgram_1s.png'), [tv_pyr, tv_bc, tv_olm, tv_sca], [fv_pyr, fv_bc, fv_olm, fv_sca], [pspec_pyr, pspec_bc, pspec_olm, pspec_sca], ["pyramidal cells", "basket cells", "olm cells", "schaffer collaterals"], xlim=[3, 4], ylim=[0, 150], **git_kwargs)

    # save vectors
    save_membrane_potential(os.path.join(dirs['data'], 'CA1_pyr_Vm.npz'), np.array(t_vec), potential_pyr) 
    save_membrane_potential(os.path.join(dirs['data'], 'CA1_bc_Vm.npz'), np.array(t_vec), potential_bc) 
    save_membrane_potential(os.path.join(dirs['data'], 'CA1_olm_Vm.npz'), np.array(t_vec), potential_olm) 
    save_membrane_potential(os.path.join(dirs['data'], 'CA3_sca_Vm.npz'), np.array(t_vec), potential_sca) 
    save_membrane_potential(os.path.join(dirs['data'], 'CA3_sca_last_Vm.npz'), np.array(t_vec), potential_sca_last) 

    save_membrane_potential(os.path.join(dirs['data'], 'CA1_pyr_i.npz'), np.array(t_vec), current_pyr) 
    save_membrane_potential(os.path.join(dirs['data'], 'CA1_bc_i.npz'), np.array(t_vec), current_bc) 
    save_membrane_potential(os.path.join(dirs['data'], 'CA1_olm_i.npz'), np.array(t_vec), current_olm) 
    save_membrane_potential(os.path.join(dirs['data'], 'CA3_sca_i.npz'), np.array(t_vec), current_sca)
    save_membrane_potential(os.path.join(dirs['data'], 'CA3_sca_last_i.npz'), np.array(t_vec), current_sca_last) 

    save_membrane_potential(os.path.join(dirs['data'], 'theta_inputs.npz'), np.array(t_vec), inputs_sca)  

    np.savez(os.path.join(dirs['data'], 'CA1_pyr_spikemon.npz'), cell_id=np.array(id_spikes_pyr), t_spike=np.array(t_spikes_pyr))
    np.savez(os.path.join(dirs['data'], 'CA1_bc_spikemon.npz'), cell_id=np.array(id_spikes_bc), t_spike=np.array(t_spikes_bc))
    np.savez(os.path.join(dirs['data'], 'CA1_olm_spikemon.npz'), cell_id=np.array(id_spikes_olm), t_spike=np.array(t_spikes_olm))
    np.savez(os.path.join(dirs['data'], 'CA3_sca_spikemon.npz'), cell_id=np.array(id_spikes_sca), t_spike=np.array(t_spikes_sca))
    np.savez(os.path.join(dirs['data'], 'CA3_sca_last_spikemon.npz'), cell_id=np.array(id_spikes_sca_last), t_spike=np.array(t_spikes_sca_last))

    # stim_loc_pyr = np.abs(ca1_pyr_coords[:,5] - settings.stim_pos[0]).argmin()
    # stim_loc_bc = np.abs(ca1_bc_coords[:,5] - settings.stim_pos[0]).argmin()

    save_raster(os.path.join(dirs['figures'], 'raster_plot.png'), [t_spikes_pyr, t_spikes_bc, t_spikes_olm, t_spikes_sca_last],
                    [id_spikes_pyr, id_spikes_bc, id_spikes_olm, id_spikes_sca_last], 
                    ['C0', 'C3', 'C1', 'C2'], ['pyramidal cells', 'basket cells', 'olm cells', 'schaffer collaterals'],
                    x_lim=[0, duration],
                    stim_time=settings.stim_onset, stim_dur=settings.stim_dur, #stim_loc=[stim_loc_pyr, n_pyr_ca1+stim_loc_bc],
                    **git_kwargs)
    
    save_raster(os.path.join(dirs['figures'], 'raster_plot_last_second.png'), [t_spikes_pyr, t_spikes_bc, t_spikes_olm, t_spikes_sca_last],
                [id_spikes_pyr, id_spikes_bc, id_spikes_olm, id_spikes_sca_last], 
                ['C0', 'C3', 'C1', 'C2'], ['pyramidal cells', 'basket cells', 'olm cells', 'schaffer collaterals'],
                x_lim=[duration - 1000, duration], size_raster=1., 
                stim_time=settings.stim_onset, stim_dur=settings.stim_dur, #stim_loc=[stim_loc_pyr, n_pyr_ca1+stim_loc_bc],
                **git_kwargs) # last second

    # np.savez(os.path.join(dirs['data'], 'theta_inputs.npz'), t=np.array(t_vec), amplitude=np.array(osc_inputs))
    np.savez(os.path.join(dirs['data'], 'i_noise.npz'), t=np.array(t_vec), noise=np.array(i_noise))

    # fig2, ax2 = plt.subplots(1,1,figsize=(9,3))
    # ax2.plot(np.array(t_vec), np.array(osc_amp), color='red')
    # ax2.set_xlabel("Time (ms)")
    # ax2.set_ylabel("nA")
    # plot_watermark(fig2, **git_kwargs)
    # plt.savefig(os.path.join(dirs['figures'], 'theta_input.png'), bbox_inches="tight")

    fig3, ax3 = plt.subplots(1,1,figsize=(9,3))
    ax3.plot(np.array(t_vec), np.array(i_noise), color='black')
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("nA")
    plot_watermark(fig3, **git_kwargs)
    plt.savefig(os.path.join(dirs['figures'], 'i_noise.png'), bbox_inches="tight")
    plt.clf()
    plt.close()

    # fig4, axs = plt.subplots(3,1,figsize=(9,3), sharex=True)
    # axs[0].plot(np.array(t_vec), np.array(osc_inputs[0]), color='red')
    # axs[0].set_ylabel("nA")
    # axs[0].set_title('Input')
    # axs[1].plot(np.array(t_vec), np.array(ca3_schaffers[0].soma_v), color='grey')
    # axs[1].set_ylabel("mV")
    # axs[1].set_title('First node')
    # axs[2].plot(np.array(t_vec), np.array(ca3_schaffers[0].last_node_v), color='black')
    # axs[2].set_xlabel("Time (ms)")
    # axs[2].set_ylabel("mV")
    # axs[2].set_title('Last node')
    # plt.setp([axs[1], axs[2]], ylim=(-90, 40))
    # plt.tight_layout()
    # fig4.suptitle('Schaffer[0]')
    # plot_watermark(fig4, **git_kwargs)
    # plt.savefig(os.path.join(dirs['figures'], 'schaffer0_mV.png'), bbox_inches="tight")
    # plt.clf()
    # plt.close()

    # fig5, axs = plt.subplots(3,1,figsize=(9,3), sharex=True)
    # axs[0].plot(np.array(t_vec), np.array(osc_inputs[0]), color='red')
    # axs[0].set_ylabel("nA")
    # axs[0].set_title('Input')
    # axs[1].plot(np.array(t_vec), np.array(ca3_schaffers[0].soma_v), color='grey')
    # axs[1].set_ylabel("mV")
    # axs[1].set_title('First node')
    # axs[2].plot(np.array(t_vec), np.array(ca3_schaffers[0].last_node_v), color='black')
    # axs[2].set_xlabel("Time (ms)")
    # axs[2].set_ylabel("mV")
    # axs[2].set_title('Last node')
    # plt.setp([axs[1], axs[2]], xlim=(180,200), ylim=(-90, 40))
    # plt.tight_layout()
    # fig5.suptitle('Schaffer[0]')
    # plot_watermark(fig5, **git_kwargs)
    # plt.savefig(os.path.join(dirs['figures'], 'schaffer0_mV_zoomed.png'), bbox_inches="tight")
    # plt.clf()
    # plt.close()

    # fig6, axs = plt.subplots(3,1,figsize=(9,3), sharex=True)
    # axs[0].plot(np.array(t_vec), np.array(osc_inputs[2]), color='red')
    # axs[0].set_ylabel("nA")
    # axs[0].set_title('Input')
    # axs[1].plot(np.array(t_vec), np.array(ca3_schaffers[2].soma_v), color='grey')
    # axs[1].set_ylabel("mV")
    # axs[1].set_title('First node')
    # axs[2].plot(np.array(t_vec), np.array(ca3_schaffers[2].last_node_v), color='black')
    # axs[2].set_xlabel("Time (ms)")
    # axs[2].set_ylabel("mV")
    # axs[2].set_title('Last node')
    # plt.setp([axs[1], axs[2]], ylim=(-90, 40))
    # plt.tight_layout()
    # fig6.suptitle('Schaffer[2]')
    # plot_watermark(fig6, **git_kwargs)
    # plt.savefig(os.path.join(dirs['figures'], 'schaffer2_mV.png'), bbox_inches="tight")
    # plt.clf()
    # plt.close()

    # fig7, axs = plt.subplots(3,1,figsize=(9,3), sharex=True)
    # axs[0].plot(np.array(t_vec), np.array(osc_inputs[2]), color='red')
    # axs[0].set_ylabel("nA")
    # axs[0].set_title('Input')
    # axs[1].plot(np.array(t_vec), np.array(ca3_schaffers[2].soma_v), color='grey')
    # axs[1].set_ylabel("mV")
    # axs[1].set_title('First node')
    # axs[2].plot(np.array(t_vec), np.array(ca3_schaffers[2].last_node_v), color='black')
    # axs[2].set_xlabel("Time (ms)")
    # axs[2].set_ylabel("mV")
    # axs[2].set_title('Last node')
    # plt.setp([axs[1], axs[2]], xlim=(180,200), ylim=(-90, 40))
    # plt.tight_layout()
    # fig7.suptitle('Schaffer[2]')
    # plot_watermark(fig7, **git_kwargs)
    # plt.savefig(os.path.join(dirs['figures'], 'schaffer2_mV_zoomed.png'), bbox_inches="tight")
    # plt.clf()
    # plt.close()

    # fig8, axs = plt.subplots(3,1,figsize=(9,3), sharex=True)
    # axs[0].plot(np.array(t_vec), np.array(osc_inputs[19]), color='red')
    # axs[0].set_ylabel("nA")
    # axs[0].set_title('Input')
    # axs[1].plot(np.array(t_vec), np.array(ca3_schaffers[19].soma_v), color='grey')
    # axs[1].set_ylabel("mV")
    # axs[1].set_title('First node')
    # axs[2].plot(np.array(t_vec), np.array(ca3_schaffers[19].last_node_v), color='black')
    # axs[2].set_xlabel("Time (ms)")
    # axs[2].set_ylabel("mV")
    # axs[2].set_title('Last node')
    # plt.setp([axs[1], axs[2]], ylim=(-90, 40))
    # plt.tight_layout()
    # fig8.suptitle('Schaffer[19]')
    # plot_watermark(fig8, **git_kwargs)
    # plt.savefig(os.path.join(dirs['figures'], 'schaffer19_mV.png'), bbox_inches="tight")
    # plt.clf()
    # plt.close()

    # fig9, axs = plt.subplots(3,1,figsize=(9,3), sharex=True)
    # axs[0].plot(np.array(t_vec), np.array(osc_inputs[19]), color='red')
    # axs[0].set_ylabel("nA")
    # axs[0].set_title('Input')
    # axs[1].plot(np.array(t_vec), np.array(ca3_schaffers[19].soma_v), color='grey')
    # axs[1].set_ylabel("mV")
    # axs[1].set_title('First node')
    # axs[2].plot(np.array(t_vec), np.array(ca3_schaffers[19].last_node_v), color='black')
    # axs[2].set_xlabel("Time (ms)")
    # axs[2].set_ylabel("mV")
    # axs[2].set_title('Last node')
    # plt.setp([axs[1], axs[2]], xlim=(180,200), ylim=(-90, 40))
    # plt.tight_layout()
    # fig9.suptitle('Schaffer[19]')
    # plot_watermark(fig9, **git_kwargs)
    # plt.savefig(os.path.join(dirs['figures'], 'schaffer19_mV_zoomed.png'), bbox_inches="tight")
    # plt.clf()
    # plt.close()

    print("[+] Saving data done !")
    print("[+] Resizing vectors")
    sys.stdout.flush()

# wait for all processors to reach this stage
pc.barrier()
pc.done()
h.quit()



