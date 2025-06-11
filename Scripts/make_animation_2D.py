import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.autonotebook import tqdm

from Scripts.svg import *
from Scripts.anatomy import *
from Model import settings
import parameters


results_dir = os.path.join(parent_dir, 'extra_stim_cartesian_results', '2025_06_06 11H31M42 2_mA new coords - inner stim - 100 ms')
data_dir = os.path.join(results_dir, 'data')
SVG_PATH = "./svg_files/"

filename = os.path.join(parent_dir, 'configs', 'parameters_inner_stim_CA1.json')

try:
    data = parameters.load(filename)
    print('Using "{0}"'.format(filename))
except Exception as e:
    print(e)
    print('Using "parameters_inner_stim_CA1.json"')
    data = parameters._data
parameters.dump(data) 
print()

# Settings initialization
settings.init(data)

# get neuron coords
positions_dir = os.path.join(parent_dir, 'positions_correct_layers_thickness', 'ca1')
pyr_coords = np.load(os.path.join(positions_dir, 'pyr_coordinates.npy'))
bc_coords = np.load(os.path.join(positions_dir, 'bc_coordinates.npy'))
olm_coords = np.load(os.path.join(positions_dir, 'olm_coordinates.npy'))

pyr_coords = pyr_coords[pyr_coords[:,5].argsort()]
bc_coords = bc_coords[bc_coords[:,5].argsort()]
olm_coords = olm_coords[olm_coords[:,5].argsort()]

stim_pos = settings.stim_pos

# get spike monitors
pyr_spikes = np.load(os.path.join(data_dir, 'CA1_pyr_spikemon.npz'))
pyr_spk_id = pyr_spikes['cell_id']
pyr_spk_t = pyr_spikes['t_spike']


bc_spikes = np.load(os.path.join(data_dir, 'CA1_bc_spikemon.npz'))
bc_spk_id = bc_spikes['cell_id']
bc_spk_t = bc_spikes['t_spike']

olm_spikes = np.load(os.path.join(data_dir, 'CA1_olm_spikemon.npz'))
olm_spk_id = olm_spikes['cell_id']
olm_spk_t = olm_spikes['t_spike']

# animation

regions = ['EC', 'Sub', 'CA1', 'CA3', 'DG']
#regions_pcs_ratios = [15, 15, 31, 8, 31]
regions_pcs_ratios = [16, 12, 29, 6, 37] # literature
regions_inh_ratios = [10, 10, 10, 10, 4]
regions_size = [20, 31, 19, 16, 14]
regions_idx = [0]
total_length = 0
for i in range(len(regions_size)):
    total_length += regions_size[i]
    regions_idx.append(total_length)

verts, codes, skeleton_points, nodes = convert(get(SVG_PATH + "hpf_update.svg", "skeleton"))

converted_verts = np.zeros(np.shape(verts))
for i in range(len(verts)):
    converted_verts[i][0] = convert_to_μm(verts[i][0])

for i in range(len(verts)):
    converted_verts[i][1] = convert_to_μm(verts[i][1])

T = np.arange(0, 1.01, 0.01)

points = []
for t in T:
    points.extend(stack_bezier(t,converted_verts[:4], converted_verts[3:]))
points = np.array(points).reshape(-1, 2)

new_points = []
for t in T:
    new_points.extend(mapping_bezier(t, points, converted_verts))
new_points = np.array(new_points).reshape(-1, 2)

thickness = 1.5*1e3

c1, d1, idx = external_shape(new_points, thickness/2)

shape_length, arcs_lengths = get_shape_length(new_points)

color = ['blue', 'brown', 'red', 'purple', 'green']
cell_colors = [[0., 0., 1., 1.], [1., 0., 0., 1.], [1., 0.54901961, 0., 1.], [0., 0., 0., 1.]]

min_size = 30
max_size = 400

# initial states
pyr_facecolors = [cell_colors[0]] * len(pyr_coords)
# pyr_facecolors = [[1., 1., 1., 1.]] * len(pyr_coords)
pyr_facecolors = np.array(pyr_facecolors)

bc_facecolors = [cell_colors[1]] * len(bc_coords)
# bc_facecolors = [[1., 1., 1., 1.]] * len(bc_coords)
bc_facecolors = np.array(bc_facecolors)

olm_facecolors = [cell_colors[2]] * len(olm_coords)
# olm_facecolors = [[1., 1., 1., 1.]] * len(olm_coords)
olm_facecolors = np.array(olm_facecolors)

pyr_sizes = [1.] * len(pyr_coords)
pyr_sizes = np.array(pyr_sizes)

bc_sizes = [1.] * len(bc_coords)
bc_sizes = np.array(bc_sizes)

olm_sizes = [1.] * len(olm_coords)
olm_sizes = np.array(olm_sizes)

elec_facecolors = np.array([[1., 1., 1., 1.]])

## set figure
fig, ax = plt.subplots(figsize=(15, 15))

ax.plot(c1[regions_idx[2]:regions_idx[3]+1,0], c1[regions_idx[2]:regions_idx[3]+1,1], color=color[2])
ax.plot(d1[regions_idx[2]:regions_idx[3]+1,0], d1[regions_idx[2]:regions_idx[3]+1,1], color=color[2])
ax.plot([d1[regions_idx[2], 0], c1[regions_idx[2], 0]], [d1[regions_idx[2], 1], c1[regions_idx[2], 1]], color=color[2])
ax.plot([d1[regions_idx[3], 0], c1[regions_idx[3], 0]], [d1[regions_idx[3], 1], c1[regions_idx[3], 1]], color=color[2])

## set parameters for animation
dt = 1
T = settings.duration
tv = np.arange(0, T, dt)
stim_on = settings.stim_onset
stim_dur  = settings.stim_dur
Twin = T - stim_on # 500ms window to animate, around the stimulation pulse
twin_idx = int(Twin/dt) # samples

pyr_raster = np.zeros([len(tv), len(pyr_coords)], dtype=bool)
bc_raster = np.zeros([len(tv), len(bc_coords)], dtype=bool)
olm_raster = np.zeros([len(tv), len(olm_coords)], dtype=bool)
elec_raster = np.zeros([len(tv), 1], dtype=bool)

for idx in range(len(pyr_spk_t)):
    # print(int(pyr_spk_id[idx]))
    pyr_raster[int(pyr_spk_t[idx]), int(pyr_spk_id[idx])] = True

for idx in range(len(bc_spk_t)):
    bc_raster[int(bc_spk_t[idx]), int(len(pyr_coords)-bc_spk_id[idx])] = True

for idx in range(len(olm_spk_t)):
    olm_raster[int(olm_spk_t[idx]), int(len(pyr_coords)+len(bc_coords)-olm_spk_id[idx])] = True

elec_raster[int(stim_on):int(stim_on+stim_dur),0] = True

pyr_scat_fix = ax.scatter(pyr_coords[:,0], pyr_coords[:,1], s=1, edgecolors=cell_colors[0], facecolors=cell_colors[0])
bc_scat_fix = ax.scatter(bc_coords[:,0], bc_coords[:,1], s=1, edgecolors=cell_colors[1], facecolors=cell_colors[1])
olm_scat_fix = ax.scatter(olm_coords[:,0], olm_coords[:,1], s=1, edgecolors=cell_colors[2], facecolors=cell_colors[2])

pyr_scat = ax.scatter(pyr_coords[:,0], pyr_coords[:,1], s=pyr_sizes, edgecolors=pyr_facecolors, facecolors=pyr_facecolors)
bc_scat = ax.scatter(bc_coords[:,0], bc_coords[:,1], s=bc_sizes, edgecolors=bc_facecolors, facecolors=bc_facecolors)
olm_scat = ax.scatter(olm_coords[:,0], olm_coords[:,1], s=olm_sizes, edgecolors=olm_facecolors, facecolors=olm_facecolors)
elec_scat = ax.scatter(settings.stim_pos[0], settings.stim_pos[1], s=400, edgecolors=cell_colors[3], facecolors=elec_facecolors)

pyr_scat.set_alpha = 0.
bc_scat.set_alpha = 0.
olm_scat.set_alpha = 0.

check_text = ax.text(settings.stim_pos[0] - 2500, settings.stim_pos[1] + 2500, 'time = 0 ms', fontsize=12)

## set update function for animation
## when spiketime -> filled
## when not -> empty
def update(frame):
    check_text.set_text(f'time = {frame * 1} ms')

    # set update for electrode
    if frame * 1 >= stim_on and frame * 1 < stim_on + stim_dur:
            elec_facecolors[0] = cell_colors[3] 
            elec_scat.set_facecolors(elec_facecolors)
    else:
        elec_facecolors[0] = [1., 1., 1., 1.]
        elec_scat.set_facecolors(elec_facecolors)

    ## sizes
    # pyr_sizes = pyr_scat.get_sizes()
    # bc_sizes = pyr_scat.get_sizes()
    # olm_sizes = pyr_scat.get_sizes()

    index_pyr = np.argwhere(pyr_sizes[(pyr_sizes > 1)] < max_size)
    index_bc = np.argwhere(bc_sizes[(bc_sizes > 1)] < max_size)
    index_olm = np.argwhere(olm_sizes[(olm_sizes > 1)] < max_size)

    pyr_sizes[index_pyr] += (max_size - min_size)/50
    bc_sizes[index_bc] += (max_size - min_size)/50
    olm_sizes[index_olm] += (max_size - min_size)/50

    pyr_facecolors[index_pyr,3] = np.maximum(0, pyr_facecolors[index_pyr,3] - 1/50)
    bc_facecolors[index_bc,3] = np.maximum(0, bc_facecolors[index_bc,3] - 1/50)
    olm_facecolors[index_olm,3] = np.maximum(0, olm_facecolors[index_olm,3] - 1/50)

    index_pyr = pyr_sizes == max_size
    index_bc = bc_sizes == max_size
    index_olm = olm_sizes == max_size

    pyr_sizes[index_pyr] = 1
    bc_sizes[index_bc] = 1
    olm_sizes[index_olm] = 1

    pyr_facecolors[index_pyr,3] = 0.
    bc_facecolors[index_bc,3] = 0.
    olm_facecolors[index_olm,3] = 0.

    # update neurons
    # get the indexes of the neurons active during frame
    # update those neurons' facecolors
    # put everything to inactive first
    # put the updated to active
    # pyramidal cells
    # pyr_facecolors[:,3] =  0.
    # pyr_facecolors[:] =  [1., 1., 1., 1.]
    index_active_pyr = np.argwhere(pyr_spk_t.astype(int) == frame * 1)
    if len(index_active_pyr) > 0:
        id_active_pyr = pyr_spk_id[index_active_pyr]
        pyr_facecolors[id_active_pyr.astype(int),3] =  1.
        pyr_sizes[id_active_pyr.astype(int)] = min_size
        # pyr_facecolors[id_active_pyr.astype(int)] = cell_colors[0] 
    pyr_scat.set_facecolors(pyr_facecolors)
    pyr_scat.set_edgecolors(pyr_facecolors)
    
    # bc cells
    # bc_facecolors[:,3] =  0.
    # bc_facecolors[:] =  [1., 1., 1., 1.]
    index_active_bc = np.argwhere(bc_spk_t.astype(int) == frame * 1)
    if len(index_active_bc) > 0:
        id_active_bc = bc_spk_id[index_active_bc]
        bc_facecolors[id_active_bc.astype(int) - len(pyr_coords),3] =  1.
        bc_sizes[id_active_bc.astype(int) - len(pyr_coords)] = min_size
        # bc_facecolors[id_active_bc.astype(int) - len(pyr_coords)] = cell_colors[1] 
    bc_scat.set_facecolors(bc_facecolors)
    bc_scat.set_edgecolors(bc_facecolors)

    # olm cells
    # olm_facecolors[:,3] =  0.
    # olm_facecolors[:] =  [1., 1., 1., 1.]
    index_active_olm = np.argwhere(olm_spk_t.astype(int) == frame * 1)
    if len(index_active_olm) > 0:
        id_active_olm = olm_spk_id[index_active_olm]
        olm_facecolors[id_active_olm.astype(int) - len(pyr_coords) - len(bc_coords),3] =  1.
        olm_sizes[id_active_olm.astype(int) - len(pyr_coords) - len(bc_coords)] = min_size
        # olm_facecolors[id_active_olm.astype(int) - len(pyr_coords) - len(bc_coords)] = cell_colors[2] 
    olm_scat.set_facecolors(olm_facecolors)
    olm_scat.set_edgecolors(olm_facecolors)

    pyr_scat.set_sizes(pyr_sizes)
    bc_scat.set_sizes(bc_sizes)
    olm_scat.set_sizes(olm_sizes)

    return elec_scat, check_text, pyr_scat, bc_scat, olm_scat,

ax.axis('off')
ani = animation.FuncAnimation(fig=fig, func=update, frames=int(settings.duration/1), interval=1)
# ani.save(os.path.join(results_dir, 'activation_20fps.gif'), fps=20)
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=20)
bar = tqdm(total=int(settings.duration/1), file=sys.stdout)
FFwriter = animation.FFMpegWriter(fps=20)
ani.save(os.path.join(results_dir, 'activation_alpha_size_20fps.mp4'), writer=FFwriter, dpi=300, progress_callback = lambda i, n: bar.update(1))
bar.close()
plt.show()

