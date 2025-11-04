import numpy as np
from neuron import h


def set_rx_point_elec(cell, stim_pos, rho):
	for sec_id in cell.all:
		if h.ismembrane('xtra', sec=sec_id):
			for seg in sec_id:
				r = np.sqrt((seg.x_xtra - stim_pos[0])**2 + (seg.y_xtra - stim_pos[1])**2 + (seg.z_xtra - stim_pos[2])**2)
				r = max(r, (400))
				# r = max(r, seg.diam/2)
				# if r==0:
				# 	r = seg.diam / 2
				
				seg.rx_xtra = (rho / (4 * np.pi * r)) * 0.01


def set_rx_bipolar(cell, stim_pos_1, stim_pos_2, rho):
	# shape_coords = len(stim_pos_1)

	# dist_elec = 0
	# for i in range(shape_coords):
	# 	dist_elec += (stim_pos_1[i] - stim_pos_2[i])**2
	# dist_elec = np.sqrt(dist_elec)

	for sec_id in cell.all:
		if h.ismembrane('xtra', sec=sec_id):
			for seg in sec_id:
				r1 = np.sqrt((seg.x_xtra - stim_pos_1[0])**2 + (seg.y_xtra - stim_pos_1[1])**2 + (seg.z_xtra - stim_pos_1[2])**2)
				r2 = np.sqrt((seg.x_xtra - stim_pos_2[0])**2 + (seg.y_xtra - stim_pos_2[1])**2 + (seg.z_xtra - stim_pos_2[2])**2)
				# r = max(r, (800))
				# r1 = max(r1, seg.diam/2)
				r1 = max(r1, 400)
				# r2 = max(r2, seg.diam/2)
				r2 = max(r2, 400)
				# if r==0:
				# 	r = seg.diam / 2
				
				seg.rx_xtra = (rho / (4 * np.pi)) * ((1/r1) - (1/r2)) * 0.01


def set_rx_bipolar_2(cell, stim_pos_1, stim_pos_2, rho): 
	# shape_coords = len(stim_pos_1)

	# dist_elec = 0
	# for i in range(shape_coords):
	# 	dist_elec += (stim_pos_1[i] - stim_pos_2[i])**2
	# dist_elec = np.sqrt(dist_elec)

	for sec_id in cell.all:
		if h.ismembrane('xtra', sec=sec_id):
			for seg in sec_id:
				r1 = np.sqrt((seg.x_xtra - stim_pos_1[0])**2 + (seg.y_xtra - stim_pos_1[1])**2 + (seg.z_xtra - stim_pos_1[2])**2)
				r2 = np.sqrt((seg.x_xtra - stim_pos_2[0])**2 + (seg.y_xtra - stim_pos_2[1])**2 + (seg.z_xtra - stim_pos_2[2])**2)
				# r = max(r, (800))
				# r1 = max(r1, seg.diam/2)
				r1 = max(r1, 400)
				# r2 = max(r2, seg.diam/2)
				r2 = max(r2, 400)

				rx1 = (rho / (4 * np.pi * r1)) * 0.01
				rx2 = (rho / (4 * np.pi * r2)) * 0.01
				# if r==0:
				# 	r = seg.diam / 2
				
				seg.rx_xtra = rx1 + rx2


def set_xtra_mechanism(cell):
    # insert xtra mechanism in all sections
    for section in cell.all:
        if not h.ismembrane('xtra', sec=section):
            section.insert('xtra')

        if not h.ismembrane('extracellular', sec=section):
            section.insert('extracellular')

        cell._grindway()

    # connect extracellular mechanism variables to the equivalent ones from xtra mechanism
    for section in cell.all:
        section.push()
        for seg in section:
            h.setpointer(seg._ref_i_membrane, 'im', seg.xtra)
            h.setpointer(seg._ref_e_extracellular, 'ex', seg.xtra)
        h.pop_section()
				

def attach_stim(cell, attached, stim_amp, stim_time):
	for sec_id in cell.all:
		if attached == 0:
			if h.ismembrane('xtra', sec=sec_id):
				stim_amp.play(h._ref_is_xtra, stim_time, True)
				attached = 1
				

# create rectangular waveform stimulation
def stim_waveform(stim_amp, stim_time, delay, dur, amp, dur_stim):
	stim_amp.resize(6)
	stim_amp.fill(0)
	stim_amp.x[2] = 1
	stim_amp.x[3] = 1
	stim_amp.mul(amp)
	
	stim_time.resize(6)
	stim_time.x[1] = delay
	stim_time.x[2] = delay
	stim_time.x[3] = delay + dur
	stim_time.x[4] = delay + dur
	stim_time.x[5] = dur_stim
	
	return stim_amp, stim_time