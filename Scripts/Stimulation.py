import numpy as np
from neuron import h


def set_rx(cell, stim_pos, rho):
	for sec_id in cell.all:
		if h.ismembrane('xtra', sec=sec_id):
			for seg in sec_id:
				r = np.sqrt((seg.x_xtra - stim_pos[0])**2 + (seg.y_xtra - stim_pos[1])**2 + (seg.z_xtra - stim_pos[2])**2)
				if r==0:
					r = seg.diam / 2
				
				seg.rx_xtra = (rho / (4 * np.pi * r)) * 0.01
				

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
def stim_waveform(stim_amp, stim_time, delay, dur, amp):
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
	stim_time.x[5] = delay + dur + 1
	
	return stim_amp, stim_time