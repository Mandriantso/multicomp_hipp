from neuron import h
from Scripts.anatomy import *

import random


def extend_axon(cell_pre, cell_pre_coords, cell_post, cell_post_coords, paced_curve, is_pre_EC=False):
    if not is_pre_EC:
        axon_depth = get_y_axonal_extension(int(cell_pre_coords[0]*100), cell_pre.axon, paced_curve)
        pre_axonal_extension = build_axonal_extension(paced_curve, axon_depth)
        pre_axonal_extension = pre_axonal_extension[math.floor(cell_post_coords[0]*100)+1:math.floor(cell_pre_coords[0]*100)]
        # extending basic shape
        for j in range(len(pre_axonal_extension)):
            h.pt3dadd(np.flip(pre_axonal_extension, axis=0)[j, 0], np.flip(pre_axonal_extension, axis=0)[j, 1], cell_pre.axon.z3d(1), 1, sec=cell_pre.axon)
        h.pt3dadd(cell_post.radTprox.x3d(1), cell_post.radTprox.y3d(1), cell_post.radTprox.z3d(1), 1, sec=cell_pre.axon)

    else:
        h.pt3dadd(cell_pre.axon.x3d(1), cell_post.axon.y3d(1), cell_post.radTprox.z3d(1), 1, sec=cell_pre.axon)
        h.pt3dadd(cell_post.radTprox.x3d(1), cell_post.radTprox.y3d(1), cell_post.radTprox.z3d(1), 1, sec=cell_pre.axon)

    # update nseg
    cell_pre.axon.nseg = 1 + 2*int(cell_pre.axon.L/40)
    cell_pre.update_shape()
    print(cell_pre.axon.L)


def get_inputs(n_inputs, inputs, cells):
    num_picks = []
    # num_picks = [i for i in range(n_inputs)]
    while len(num_picks) < n_inputs:
        n_ = random.randint(0, len(cells)-1)
        if n_ not in num_picks:
            num_picks.append(n_)

    for ns, i in zip(inputs[:n_inputs], num_picks):
        cells[i].inputs_list.append(ns)

    return num_picks


# def get_post_cells(cell, list_post_cells):
#     for i in range(len(list_post_cells)):
#         dist_value = math.sqrt((list_post_cells[i].x - cell.x)**2 + (list_post_cells[i].y - cell.y)**2 + (list_post_cells[i].z - cell.z)**2)
#         if dist_value <= cell.syn_dist and list_post_cells[i] not in cell.post_list and list_post_cells[i] != cell:
#             cell.post_list.append(list_post_cells[i])

def get_post_cells(pc, cell, list_cellpostgids):
    for cellpostgid_ in list_cellpostgids:
        dist_value = math.sqrt((pc.gid2cell(cellpostgid_).x - cell.x)**2 + (pc.gid2cell(cellpostgid_).y - cell.y)**2 + (pc.gid2cell(cellpostgid_).z - cell.z)**2)
        if dist_value <= cell.syn_dist and cellpostgid_ not in cell.post_list and pc.gid2cell(cellpostgid_) != cell:
            # cell.post_list.append(pc.gid2cell(cellpostgid_))
            cell.post_list.append(cellpostgid_)


# def get_pre_cells(cell, list_pre_cells):
#     for i in range(len(list_pre_cells)):
#         if cell in list_pre_cells[i].post_list and list_pre_cells[i] not in cell.pre_list:
#             cell.pre_list.append(list_pre_cells[i])


def get_pre_cells(pc, cell, list_cellpregids):
    for cellpregid_ in list_cellpregids:
        if cell in pc.gid2cell(cellpregid_).post_list and pc.gid2cell(cellpregid_) not in cell.pre_list:
            cell.pre_list.append(cellpregid_)


def create_synapse(cell, cell_pre_section, t_rise, t_decay, e):
    syn_ = h.Exp2Syn(cell_pre_section(0.5))
    syn_.tau1 = t_rise
    syn_.tau2 = t_decay
    syn_.e = e
    cell.syn_list.append(syn_)


def connect_cells(pc, cell_pre_gid, cell_pre_section, cell_post_section, weight, threshold, delay):
    # retrieve synapse point process
    mt_ = h.MechanismType(1)
    mt_.select("Exp2Syn")
    pp = mt_.pp_begin(sec=cell_post_section)

    # create NetCon object
    # nc_ = h.NetCon(cell_pre_section(1)._ref_v, pp, sec=cell_pre_section)
    # netcon = pc.gid_connect(cell_pre_gid, pp, nc_)
    netcon = pc.gid_connect(cell_pre_gid, pp)
    netcon.weight[0] = weight
    netcon.threshold = threshold
    netcon.delay = delay
    pc.gid2cell(cell_pre_gid).nc_list.append(netcon)


def connect_inputs(input_cell, cell_post, cell_post_section, weight, delay):
    # retrieve synapse point process
    mt_ = h.MechanismType(1)
    mt_.select("Exp2Syn")
    pp = mt_.pp_begin(sec=cell_post_section)

    # create NetCon object
    nc_ = h.NetCon(input_cell, pp)
    nc_.weight[0] = weight
    nc_.delay = delay
    cell_post.nc_list_inputs.append(nc_)

    