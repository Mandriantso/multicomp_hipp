import os
import time
import sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from neuron import h
from Cells.Cells import *
from Scripts.utilities import *
from Scripts.Stimulation import *
from Scripts.myplot import plot_watermark


def plot_2D_custom(h, ax, section, scale, colors, d=0):
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
			ax.plot(xyd[i:i+2,0], xyd[i:i+2,1], color=colors[i], linewidth=scale)
		ax.plot(xyd[-2:,0], xyd[-2:,1], color=colors[-1], linewidth=scale)
	else:
		ax.plot(xyd[:,0], xyd[:,1], color=colors[0], linewidth=scale)


def make_heatmap(name, cell, thresh_array, colormap, x_array, y_array, title=None, norm=None, **kwargs):
	X, Y = np.meshgrid(x_array, y_array)
	colormap.set_bad('white')

	thresh_array[thresh_array==0.0] = np.nan
	print(thresh_array)
	fig, ax = plt.subplots(1, 1, sharey='row', figsize=(10,10))
	length = fig.bbox_inches.height * ax.get_position().height / 3
	value_range = np.diff(ax.get_ylim())
	for sec in cell.all:
		plot_2D_custom(h, ax, sec, scale=length/value_range, colors=['white'], d=1)
	for sec in cell.all:
		plot_2D_custom(h, ax, sec, scale=length/value_range, colors=['black'])
	ax.set_ylabel('Y (µm)')
	ax.set_xlabel('X (µm)')
	ax.set_xlim(x_array.min(), x_array.max())
	ax.set_ylim(y_array.min(), y_array.max())
	if norm:
		im_thr = ax.imshow(np.absolute(thresh_array.transpose()), cmap=colormap, norm=norm, origin='lower', extent=[x_array.min(), x_array.max(), y_array.min(), y_array.max()], aspect='auto', interpolation='bicubic')
	else:
		im_thr = ax.imshow(np.absolute(thresh_array.transpose()), cmap=colormap, origin='lower', extent=[x_array.min(), x_array.max(), y_array.min(), y_array.max()], aspect='auto', interpolation='bicubic')
	contours = plt.contour(X, Y, np.absolute(thresh_array.transpose()), 8, colors='azure', interpolation='bicubic')
	plt.clabel(contours, inline=1, fontsize=10)
	cbar = fig.colorbar(im_thr, ax=ax)
	cbar.set_label("Stimulus amplitude (mA)")
	if title:
		fig.suptitle(title)
	print('saving figure...')
	plot_watermark(fig, **kwargs)
	plt.show()
	# plt.savefig(name, format='png', bbox_inches='tight', pad_inches = 0)
	# plt.clf()
	# plt.close()


def run_simulation(id_pos_x, pos_x, id_pos_y, pos_y):
	AMPS_CAT = np.arange(-0.01, -10.01, -0.01)
	POS_Y = np.arange(-1000, 1050, 50)
	POS_X = np.arange(0, 850, 50)

	thresh_cat = 0
	# setting extracellular stimulation parameters
    # medium resistivity ohm cm
	RHO = 300

	# stim parameters ms
	DEL = 100 
	DUR = 1 # 0.1 # 100

	ATTACHED__ = 0
	print('Creating new cell...\n')
	# create new cell
	cell = PyramidalCell(gid_soma=0, gid_axon=1)
	# cell = BasketCell(gid=pc.id())
	# cell = OLMCell(gid=pc.id())
	# cell = PyrCell(0)

	print('Setting xtra mechanism...\n')
	# insert xtra mechanism in all sections
	# and connect extracellular mechanism variables to the equivalent ones from xtra
	set_xtra_mechanism(cell)

	print('Fixing nonvariable stimulation parameters...\n')
	h.dt = 0.025 # ms
	#h.tstop = 350 # 350
	h.v_init = -65
	h.celsius = 35

	# create stimulus vectors
	stim_amp = h.Vector()
	stim_time = h.Vector()

	# create recording vectors
	t_vec = h.Vector()

	thresh_soma = 0

	X_DIR = os.path.join(SAVE_DIR, 'x_{}_µm'.format(pos_x))
	if not os.path.exists(X_DIR):
		os.mkdir(X_DIR)

	Y_DIR = os.path.join(X_DIR, 'y_{}_µm'.format(pos_y))
	if not os.path.exists(Y_DIR):
		os.mkdir(Y_DIR)

	for amp in AMPS_CAT:
		AMP_DIR = os.path.join(Y_DIR, '{}_mA'.format(amp))
		if not os.path.exists(AMP_DIR):
			os.mkdir(AMP_DIR)

		print('-----------------------------------------------------------------------')
		print('                     ( x : {}, y : {}, amp : {:.2f} )                      '.format(pos_x, pos_y, amp))
		print('-----------------------------------------------------------------------\n\n\n')

		stim_pos = (pos_x, pos_y,0)
		print('Setting recording vectors...')
		t_vec.record(h._ref_t)

		set_rx_point_elec(cell, stim_pos, RHO)
		stim_amp, stim_time = single_pulse(stim_amp, stim_time, DEL, DUR, round(amp, 2), dur_stim=350)
		attach_stim(cell, ATTACHED__, stim_amp, stim_time)

		print('Running simulation...')
		# run simulation
		h.init()
		h.finitialize(-65)
		h.cvode_active(0)
		start_time = time.time()
		h.continuerun(350)
		end_time = time.time()

		hours, rem = divmod(end_time - start_time, 3600)
		minutes, seconds = divmod(rem, 60)
		print("Elapsed time : {:0>2} h {:0>2} min {:05.2f} s".format(int(hours),int(minutes),seconds))

		print('Saving vectors...')
		np.savez(os.path.join(AMP_DIR, 'mV_soma.npz'), time=np.array(t_vec), soma=np.array(cell.soma_v))
		
# 		print('Saving figure...')
# # 			# savefig mV soma
# 		fig = plt.figure(figsize=(6,6))
# 		plt.plot(t_vec, cell.soma_v, c='red')
# 		# plt.plot(t_vec, v_vec_soma_potential, c='red')
# 		plt.title('Evolution of Membrane potential in soma')
# 		plt.xlabel('Time (ms)')
# 		plt.ylabel('Membrane potential (mV)')
# 		plt.savefig(os.path.join(AMP_DIR, 'membrane_potential_soma_35deg.png'), format='png', bbox_inches='tight', pad_inches = 0)
# 		plt.clf()   
# 		plt.close()
		
		print('Updating threshold_array...')
		# check if action potential
		if (cell.soma_v.max() >= 0) and (thresh_cat==0):
			thresh_cat = round(amp, 2)
			thresh_soma = 1

		print('Resizing vectors...\n\n\n')
		# resize vectors (restart)
		t_vec.resize(0)
		cell.soma_v.resize(0)

		print('update state')
		with open(os.path.join(SAVE_DIR, "output_cat.txt"), "a") as f:
			f.write(f"pos_x : {POS_X[int(id_pos_x)]}   pos_y : {POS_Y[int(id_pos_y)]}    amp : {round(amp, 2)}  done !\n")

		# exit for loop if an AP has been detected
		if thresh_soma:
			break

	return id_pos_x, id_pos_y, thresh_cat


if __name__ == "__main__":
	h.nrnmpi_init()
	pc = h.ParallelContext()

	# print("Create directory")
	DIR = './new_heatmap_results_paper/'
	if not os.path.exists(DIR):
		os.mkdir(DIR)

	SAVE_DIR = os.path.join(DIR, 'Pyr_cell_1ms_debug')
	if not os.path.exists(SAVE_DIR):
		os.mkdir(SAVE_DIR)

	print("Set iterables")
    # setting iterables
	POS_X = np.arange(0, 850, 50)
	POS_Y = np.arange(-1000, 1050, 50)

	print("Initialize arrays")
	thresh_array_cat_soma = np.zeros((len(POS_X), len(POS_Y)))
	# thresh_array_an_soma = np.zeros((len(POS_X), len(POS_Y)))
	
	print("Start parellization")
	total = len(POS_Y) * len(POS_X)
	count = 0
	pc.runworker()

	for id_pos_y, pos_y in enumerate(POS_Y):
		if (-600 <= pos_y <= -500) or (500 <= pos_y <= 600):
			for id_pos_x, pos_x in enumerate(POS_X):
				if pos_x >= 400 and pos_x<= 500:
					pc.submit(run_simulation, id_pos_x, pos_x, id_pos_y, pos_y)
					print(f'job {count}/{total} submitted')
					count += 1

	while pc.working():
		id_pos_x, id_pos_y, thresh_cat = pc.pyret()
		thresh_array_cat_soma[int(id_pos_x), int(id_pos_y)] = thresh_cat
		np.save(os.path.join(SAVE_DIR, f'soma_cathodic_threshold_array.npy'), thresh_array_cat_soma)

	pc.barrier()
	print("all jobs done")
	pc.done()
	h.quit()
	print("save vector")
	# save thresh_array
	np.save(os.path.join(SAVE_DIR, f'soma_cathodic_threshold_array.npy'), thresh_array_cat_soma)
	sys.exit()

	# thresh_array_cat_soma = np.load(os.path.join(SAVE_DIR, "soma_cathodic_threshold_array_100ms_35deg.npy"))
	# thresh_array_cat_soma_2 = np.load(os.path.join(SAVE_DIR_2, "soma_cathodic_threshold_array_100ms_35deg.npy"))

	# thresh_array_cat_soma = thresh_array_cat_soma + thresh_array_cat_soma_2
	# thresh_array_cat_soma[:,5] = thresh_array_cat_soma[:,21]
	# print(thresh_array_cat_soma)
	# thresh_array_an_soma = np.load(os.path.join(SAVE_DIR, "soma_anodic_threshold_array_100ms_35deg.npy"))
	# thresh_array_an_soma_2 = np.load(os.path.join(SAVE_DIR_2, "soma_anodic_threshold_array_100ms_35deg.npy"))
	# thresh_array_an_soma = thresh_array_an_soma + thresh_array_an_soma_2
	# thresh_array_an_soma[:,5] = thresh_array_an_soma[:,21]
	# print(thresh_array_an_soma)

	# thr_min = min(abs(thresh_array_cat_soma.min()), thresh_array_an_soma.min())
	# thr_max = max(abs(thresh_array_cat_soma.max()), thresh_array_an_soma.max())

	# norm = matplotlib.colors.Normalize(vmin=thr_min, vmax=thr_max)

	# create new cell
	# cell = PyramidalCell(gid_soma=0, gid_axon=1)
	# cell = OLMCell(gid=0)

	# print("Make heatmap")
	# plot heatmap
	# make_heatmap(os.path.join(SAVE_DIR, 'cathodic_soma.png'), cell, thresh_array_cat_soma, matplotlib.colormaps['hot'],
	# 		   POS_X, POS_Y, 'Cathodic stimulation threshold soma', norm=norm)
	
	# make_heatmap(os.path.join(SAVE_DIR, 'anodic_soma.png'), cell, thresh_array_an_soma, matplotlib.colormaps['hot'],
	# 		   POS_X, POS_Y, 'Anodic stimulation threshold soma', norm=norm)
