import os
import sys
import time

import numpy as np
from collections import OrderedDict

import efel
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from PyNeuronToolbox.morphology import shapeplot, mark_locations

from neuron import h, hclass
from Cells.Cells import *
from Scripts.utilities import *
from Scripts.Stimulation import *
from Scripts.myplot import plot_watermark


matplotlib.use('Agg')

# h.nrn_load_dll('Mods_Tomko/nrnmech.dll')
# h.load_file("stdgui.hoc")
# h.load_file('./Cells/Tomko_ca1_pyr.hoc')

# class PyrCell(hclass(h.CA1_PC_Tomko)):
# 	def _grindway(self):
# 		for section in self.all:
# 			if h.ismembrane('xtra', sec=section):
				
# 				nn = section.n3d()
# 				xx = h.Vector(nn)
# 				yy = h.Vector(nn)
# 				zz = h.Vector(nn)
# 				length = h.Vector(nn)
				
# 				for i in range(nn):
# 					xx.x[i] = section.x3d(i)
# 					yy.x[i] = section.y3d(i)
# 					zz.x[i] = section.z3d(i)
# 					length.x[i] = section.arc3d(i)
					
# 				length.div(length.x[nn-1])
				
# 				r = h.Vector(section.nseg + 2)
# 				r.indgen(1/section.nseg)
# 				r.sub(1/(2 * section.nseg))
# 				r.x[0] = 0
# 				r.x[section.nseg + 1] = 1
				
# 				xint = h.Vector(section.nseg+2)
# 				yint = h.Vector(section.nseg+2)
# 				zint = h.Vector(section.nseg+2)
# 				xint.interpolate(r, length, xx)
# 				yint.interpolate(r, length, yy)
# 				zint.interpolate(r, length, zz)
				
# 				for i in range(1, section.nseg+1):
# 					xr = r.x[i]
# 					section(xr).x_xtra = xint.x[i]
# 					section(xr).y_xtra = yint.x[i]
# 					section(xr).z_xtra = zint.x[i]


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
		im_thr = ax.imshow(np.absolute(thresh_array.transpose()), cmap=colormap, norm=norm, origin='lower', extent=[x_array.min(), x_array.max(), y_array.min(), y_array.max()], aspect='auto', interpolation='nearest')
	else:
		im_thr = ax.imshow(np.absolute(thresh_array.transpose()), cmap=colormap, origin='lower', extent=[x_array.min(), x_array.max(), y_array.min(), y_array.max()], aspect='auto', interpolation='nearest')
	contours = plt.contour(X, Y, np.absolute(thresh_array.transpose()), 8, colors='azure', interpolation='bicubic')
	plt.clabel(contours, inline=1, fontsize=10)
	cbar = fig.colorbar(im_thr, ax=ax)
	cbar.set_label("Stimulus amplitude (mA)")
	if title:
		fig.suptitle(title)
	print('saving figure...')
	plot_watermark(fig, **kwargs)
	plt.savefig(name, format='png', bbox_inches='tight', pad_inches = 0)
	plt.clf()
	plt.close()



def main():
	start_tot = time.time()
	SAVE_DIR = './heatmap_results_paper/'
	if not os.path.exists(SAVE_DIR):
		os.mkdir(SAVE_DIR)

	# for watermaks on figures -> reproducibility
	git_kwargs = {
		'timestamp': time.ctime(),
		'branch': get_git_revision_branch(), 
		'hash': get_git_revision_hash(),
		'script_name': os.path.basename(__file__)
				}

	# setting iterables
	AMPS_CAT = np.arange(-0.01, -10.01, -0.01)
	AMPS_AN = np.arange(0.01, 10.01, 0.01)
	POS_X = np.arange(0, 850, 50)
	POS_Y = np.arange(-1000, 1050, 50)

	thresh_array_cat_soma = np.zeros((len(POS_X), len(POS_Y)))
	thresh_array_an_soma = np.zeros((len(POS_X), len(POS_Y)))

	# medium resistivity ohm cm
	RHO = 300

	# stim parameters ms
	DEL = 100 
	DUR = 100 # 0.1 # 100

	ATTACHED__ = 0

	print('Creating new cell...\n')
	# create new cell
	cell = PyramidalCell(gid_soma=0, gid_axon=1)
	# cell = BasketCell(gid=0)
	# cell = OLMCell(gid=0)
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
	# v_vec_soma_potential = h.Vector()

	print('Starting iterations...\n')
	for i, pos_x in enumerate(POS_X):
		X_DIR = os.path.join(SAVE_DIR, 'x_{}_µm'.format(pos_x))
		if not os.path.exists(X_DIR):
			os.mkdir(X_DIR)

		for j, pos_y in enumerate(POS_Y):
			Y_DIR = os.path.join(X_DIR, 'y_{}_µm'.format(pos_y))
			if not os.path.exists(Y_DIR):
				os.mkdir(Y_DIR)

			thresh_soma = 0

			for amp in AMPS_CAT:
				AMP_DIR = os.path.join(Y_DIR, '{}_mA'.format(amp))
				if not os.path.exists(AMP_DIR):
					os.mkdir(AMP_DIR)

				print('-----------------------------------------------------------------------')
				print('                     ( x : {}, y : {}, amp : {} )                      '.format(pos_x, pos_y, amp))
				print('-----------------------------------------------------------------------\n\n\n')

				stim_pos = (pos_x, pos_y,0)
				print('Setting recording vectors...')
				t_vec.record(h._ref_t)
				# v_vec_soma_potential.record(cell.soma(0.5)._ref_v)

				set_rx_point_elec(cell, stim_pos, RHO)
				stim_amp, stim_time = stim_waveform(stim_amp, stim_time, DEL, DUR, amp)
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
				np.savez(os.path.join(AMP_DIR, 'stimulation_vectors_100ms_35deg.npz'), time=np.array(t_vec), soma=np.array(cell.soma_v))
				# np.savez(os.path.join(AMP_DIR, 'stimulation_vectors_100ms_35deg.npz'), time=np.array(t_vec), soma=np.array(v_vec_soma_potential))

				print('Saving figure...')
	# 			# savefig mV soma
				fig = plt.figure(figsize=(6,6))
				plt.plot(t_vec, cell.soma_v, c='red')
				# plt.plot(t_vec, v_vec_soma_potential, c='red')
				plt.title('Evolution of Membrane potential in soma')
				plt.xlabel('Time (ms)')
				plt.ylabel('Membrane potential (mV)')
				plot_watermark(fig, **git_kwargs)
				plt.savefig(os.path.join(AMP_DIR, 'membrane_potential_soma_35deg.png'), format='png', bbox_inches='tight', pad_inches = 0)
				plt.clf()   
				plt.close()

				print('Updating threshold_array...')
				# check if action potential
				if (cell.soma_v.max() >= 0) and (thresh_array_cat_soma[i,j]==0):
				# if (v_vec_soma_potential.max() >= 0) and (thresh_array_cat_soma[i,j]==0):
					thresh_array_cat_soma[i,j] = amp
					thresh_soma = 1

				np.save(os.path.join(SAVE_DIR, 'soma_cathodic_threshold_array_100ms_35deg.npy'), thresh_array_cat_soma)


				print('Resizing vectors...\n\n\n')
				# resize vectors (restart)
				t_vec.resize(0)
				cell.soma_v.resize(0)
				# v_vec_soma_potential.resize(0)
				
				# exit for loop if an AP has been detected
				if thresh_soma:
					break

			thresh_soma = 0
			
			for amp in AMPS_AN:
				AMP_DIR = os.path.join(Y_DIR, '{}_mA'.format(amp))
				if not os.path.exists(AMP_DIR):
					os.mkdir(AMP_DIR)

				print('-----------------------------------------------------------------------')
				print('                      ( x : {},  y : {}, amp : {} )                    '.format(pos_x, pos_y, amp))
				print('-----------------------------------------------------------------------\n\n\n')

				stim_pos = (pos_x, pos_y,0)
				print('Setting recording vectors...')
				t_vec.record(h._ref_t)
				# v_vec_soma_potential.record(cell.soma(0.5)._ref_v)

				set_rx_point_elec(cell, stim_pos, RHO)
				stim_amp, stim_time = stim_waveform(stim_amp, stim_time, DEL, DUR, amp)
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
				np.savez(os.path.join(AMP_DIR, 'stimulation_vectors_100ms_35deg.npz'), time=np.array(t_vec), soma=np.array(cell.soma_v))
				# np.savez(os.path.join(AMP_DIR, 'stimulation_vectors_100ms_35deg.npz'), time=np.array(t_vec), soma=np.array(v_vec_soma_potential))

				print('Saving figure...')
	# 			# savefig mV soma
				fig = plt.figure(figsize=(6,6))
				plt.plot(t_vec, cell.soma_v, c='red')
				# plt.plot(t_vec, v_vec_soma_potential, c='red')
				plt.title('Evolution of Membrane potential in soma')
				plt.xlabel('Time (ms)')
				plt.ylabel('Membrane potential (mV)')
				plot_watermark(fig, **git_kwargs)
				plt.savefig(os.path.join(AMP_DIR, 'membrane_potential_soma_35deg.png'), format='png', bbox_inches='tight', pad_inches = 0)
				plt.clf()   
				plt.close()

				print('Updating threshold_array...')
				# check if action potential
				if (cell.soma_v.max() >= 0) and (thresh_array_an_soma[i,j]==0):
				# if (v_vec_soma_potential.max() >= 0) and (thresh_array_an_soma[i,j]==0):
					thresh_array_an_soma[i,j] = amp
					thresh_soma = 1

				np.save(os.path.join(SAVE_DIR, 'soma_anodic_threshold_array_100ms_35deg.npy'), thresh_array_an_soma)


				print('Resizing vectors...\n\n\n')
				# resize vectors (restart)
				t_vec.resize(0)
				cell.soma_v.resize(0)
				# v_vec_soma_potential.resize(0)

				if thresh_soma:
					break

	# save thresh_array
	np.save(os.path.join(SAVE_DIR, f'soma_cathodic_threshold_array_{str(DUR)}ms_{str(h.celsius)}deg.npy'), thresh_array_cat_soma)
	np.save(os.path.join(SAVE_DIR, f'soma_anodic_threshold_array_{str(DUR)}ms_{str(h.celsius)}deg.npy'), thresh_array_an_soma)

	end_tot = time.time()

	hours, rem = divmod(end_tot - start_tot, 3600)
	minutes, seconds = divmod(rem, 60)
	thr_min = min(thresh_array_cat_soma.min(), thresh_array_an_soma.min())
	thr_max = min(thresh_array_cat_soma.max(), thresh_array_an_soma.max())

	norm = matplotlib.colors.Normalize(vmin=thr_min, vmax=thr_max)

	print("Total computational time : {:0>2} h {:0>2} min {:05.2f} s".format(int(hours),int(minutes),seconds))
	# plot heatmap
	make_heatmap(os.path.join(SAVE_DIR, 'cathodic_soma.png'), cell, thresh_array_cat_soma, matplotlib.colormaps['hot'],
			   POS_X, POS_Y, 'Cathodic stimulation threshold soma', norm=norm)
	
	make_heatmap(os.path.join(SAVE_DIR, 'anodic_soma.png'), cell, thresh_array_an_soma, matplotlib.colormaps['hot'],
			   POS_X, POS_Y, 'Anodic stimulation threshold soma', norm=norm)
				

if __name__ == '__main__':
	main()
	# SAVE_DIR = './heatmap_results/'
	# if not os.path.exists(SAVE_DIR):
	# 	os.mkdir(SAVE_DIR)

	# cell = PyramidalCell(gid_soma=0, gid_axon=1)
	# POS_X = np.arange(0, 1000, 50)
	# POS_Y = np.arange(-1000, 1000, 50)

	# thresh_array_cat_soma = np.load(os.path.join(SAVE_DIR, 'soma_cathodic_threshold_array_100ms_35deg.npy'))
	# thresh_array_an_soma = np.load(os.path.join(SAVE_DIR, 'soma_anodic_threshold_array_100ms_35deg.npy'))

	# make_heatmap(os.path.join(SAVE_DIR, 'cathodic_soma_2.png'), cell, thresh_array_cat_soma, matplotlib.colormaps['hot'],
	# 		   POS_X, POS_Y, 'Cathodic stimulation threshold soma')
	
	# # make_heatmap(SAVE_DIR + 'cathodic_axon', cell, thresh_array_cat_axon500, matplotlib.colormaps['hot'],
	# # 		   POS_X, POS_Y, 'Cathodic stimulation threshold axon')
	# make_heatmap(os.path.join(SAVE_DIR, 'anodic_soma_2.png'), cell, thresh_array_an_soma, matplotlib.colormaps['hot'],
	# 		   POS_X, POS_Y, 'Anodic stimulation threshold soma')

